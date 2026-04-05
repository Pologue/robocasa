"""
Generate prefix video segments from annotated RoboCasa affordance data.

This script divides episodes into num_segments equal parts and selects optimal
split points based on target visibility, occlusion, and target jumping.

Usage:
    python generate_prefix_segments.py --outputs-dir outputs --num-segments 4 --num-workers 8
    python generate_prefix_segments.py --outputs-dir outputs --num-segments 3 --task DeliverStraw
"""

import argparse
import json
import logging
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.extract_affordance_gt import build_env
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate prefix video segments from affordance annotations."
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Root outputs directory containing task folders (e.g., outputs/).",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        required=True,
        help="Number of segments to divide each episode into.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to LeRobot dataset root for video generation. If None, only metadata is generated.",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_agentview_center",
        help="Camera name to use for segment indexing and video generation.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=512,
        help="Camera image height.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=768,
        help="Camera image width.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Process only a specific task (for debugging).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes that already have prefix segments.",
    )
    parser.add_argument(
        "--search-window",
        type=int,
        default=30,
        help="Search window (frames) around each target split point for optimal cut location.",
    )
    parser.add_argument(
        "--target-jump-threshold",
        type=float,
        default=0.05,
        help="Meters. Skip split points where target has large 3D jumps.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {path}: {e}")
    return rows


def load_dense_npz(path: Path) -> Optional[Dict]:
    """Load dense per-frame annotations from NPZ."""
    try:
        data = np.load(path, allow_pickle=True)
        return {
            "frame": data["frame"],
            "target_name": data["target_name"],
            "target_type": data["target_type"],
            "target_xyz": data["target_xyz"],
            "target_uv": data["target_uv"],
            "target_visible": data["target_visible"],
            "target_visible_pixel_count": data["target_visible_pixel_count"],
            "target_bbox": data["target_bbox"],
        }
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def compute_target_jump(xyz_prev: Optional[np.ndarray], xyz_curr: Optional[np.ndarray]) -> float:
    """Compute 3D distance between consecutive target positions."""
    if xyz_prev is None or xyz_curr is None:
        return 0.0
    if not np.all(np.isfinite(xyz_prev)) or not np.all(np.isfinite(xyz_curr)):
        return 0.0
    return float(np.linalg.norm(np.array(xyz_curr) - np.array(xyz_prev)))


def is_good_split_point(
    dense: Dict,
    frame_idx: int,
    target_jump_threshold: float,
    prefer_visible: bool = True,
) -> Tuple[bool, float]:
    """
    Evaluate if frame_idx is a good split point.
    
    Returns: (is_good, quality_score) where quality_score is used to rank candidates.
    """
    if frame_idx < 0 or frame_idx >= len(dense["target_visible"]):
        return False, -1.0

    # Check visibility
    is_visible = bool(dense["target_visible"][frame_idx])
    visible_pixels = int(dense["target_visible_pixel_count"][frame_idx])

    # Check for target jump
    jump = 0.0
    if frame_idx > 0:
        jump = compute_target_jump(
            dense["target_xyz"][frame_idx - 1],
            dense["target_xyz"][frame_idx],
        )

    # Penalize large jumps
    if jump >= target_jump_threshold:
        return False, -2.0

    # Score based on visibility
    score = 0.0
    if is_visible and visible_pixels > 50:
        score = 1.0 + visible_pixels / 1000.0
    elif prefer_visible:
        return False, -0.5
    else:
        score = 0.5

    return True, score


def find_optimal_split_point(
    dense: Dict,
    target_frame: int,
    search_window: int,
    target_jump_threshold: float,
) -> int:
    """
    Find the best split point near target_frame.
    
    Search within [target_frame - search_window, target_frame + search_window].
    Prefer frames where target is visible and no large jumps occur.
    """
    num_frames = len(dense["target_visible"])
    search_start = max(0, target_frame - search_window)
    search_end = min(num_frames - 1, target_frame + search_window)

    best_frame = target_frame
    best_score = -float("inf")

    # Try to find the best split point in the search window
    for frame_idx in range(search_start, search_end + 1):
        is_good, score = is_good_split_point(dense, frame_idx, target_jump_threshold)
        if score > best_score:
            best_score = score
            best_frame = frame_idx

    return best_frame


def compute_segment_split_frames(
    num_frames: int,
    num_segments: int,
    dense: Dict,
    search_window: int,
    target_jump_threshold: float,
) -> List[int]:
    """
    Compute split frame indices for dividing an episode into num_segments.
    
    Returns list of end frames for each segment, e.g., [100, 200, 300] for 3 segments.
    The last frame (num_frames - 1) is implied but not included in the result.
    """
    if num_segments <= 1:
        return []

    # Compute uniform target split points
    target_splits = []
    for i in range(1, num_segments):
        target_frame = int((i * num_frames) / num_segments) - 1
        target_frame = max(0, min(num_frames - 1, target_frame))
        target_splits.append(target_frame)

    # Find optimal split points near each target
    split_frames = []
    for target_frame in target_splits:
        optimal_frame = find_optimal_split_point(
            dense,
            target_frame,
            search_window,
            target_jump_threshold,
        )
        split_frames.append(optimal_frame)

    # Ensure splits are sorted and unique
    split_frames = sorted(set(split_frames))

    return split_frames


def generate_prefix_segments_for_episode(
    task_name: str,
    ep_idx: int,
    camera_name: str,
    task_dir: Path,
    num_segments: int,
    search_window: int,
    target_jump_threshold: float,
    skip_existing: bool,
) -> Dict:
    """
    Generate prefix segment metadata for a single episode.
    
    Returns dict with segment information and any errors encountered.
    """
    result = {
        "task_name": task_name,
        "episode_index": ep_idx,
        "success": False,
        "error": None,
        "segments": [],
    }

    try:
        # Build paths
        camera_dir = task_dir / camera_name
        if not camera_dir.exists():
            result["error"] = f"Camera directory not found: {camera_dir}"
            return result

        summary_path = camera_dir / "affordance_summary_masked.jsonl"
        dense_dir = camera_dir / "dense_masked"

        # Load summary
        summaries = load_jsonl(summary_path)
        if not summaries or len(summaries) == 0:
            result["error"] = "affordance_summary_masked.jsonl is empty"
            return result

        # Find summary for this episode
        ep_summary = None
        for s in summaries:
            if s.get("episode_index") == ep_idx:
                ep_summary = s
                break

        if ep_summary is None:
            result["error"] = f"Episode {ep_idx} not found in summary"
            return result

        # Check if already segmented
        if skip_existing:
            segment_dir = task_dir / f"episode_{ep_idx:06d}_prefix_{num_segments}seg"
            if segment_dir.exists():
                result["error"] = "Already segmented (skipped)"
                return result

        # Load dense annotations
        dense_path = dense_dir / f"episode_{ep_idx:06d}.npz"
        if not dense_path.exists():
            result["error"] = f"Dense file not found: {dense_path}"
            return result

        dense = load_dense_npz(dense_path)
        if dense is None:
            result["error"] = "Failed to load dense NPZ"
            return result

        num_frames = len(dense["target_visible"])

        # Compute segment split frames
        split_frames = compute_segment_split_frames(
            num_frames,
            num_segments,
            dense,
            search_window,
            target_jump_threshold,
        )

        # Build segment records
        segments = []
        start_frame = 0
        for seg_idx, end_frame in enumerate(split_frames):
            segment = {
                "segment_index": seg_idx,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "target_name": str(dense["target_name"][end_frame]),
                "target_visible_at_end": bool(dense["target_visible"][end_frame]),
                "target_visible_pixels_at_end": int(
                    dense["target_visible_pixel_count"][end_frame]
                ),
            }
            segments.append(segment)
            start_frame = end_frame + 1

        # Add final segment
        if start_frame < num_frames:
            segment = {
                "segment_index": len(split_frames),
                "start_frame": start_frame,
                "end_frame": num_frames - 1,
                "target_name": str(dense["target_name"][-1]),
                "target_visible_at_end": bool(dense["target_visible"][-1]),
                "target_visible_pixels_at_end": int(
                    dense["target_visible_pixel_count"][-1]
                ),
            }
            segments.append(segment)

        result["segments"] = segments
        result["success"] = True
        result["num_frames"] = num_frames
        result["original_task"] = ep_summary.get("task", ""),
        result["instruction"] = ep_summary.get("instruction", ""),

    except Exception as e:
        result["error"] = f"Exception: {traceback.format_exc()}"

    return result


def save_segment_metadata(task_dir: Path, ep_idx: int, num_segments: int, segments: List[Dict]):
    """Save segment metadata to a JSON file."""
    output_dir = task_dir / f"episode_{ep_idx:06d}_prefix_{num_segments}seg"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "segments_metadata.json"
    metadata = {
        "episode_index": ep_idx,
        "num_segments": len(segments),
        "segments": segments,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def process_episode(args_tuple: Tuple) -> Dict:
    """Worker function for parallelization."""
    (
        task_name,
        ep_idx,
        camera_name,
        task_dir,
        num_segments,
        search_window,
        target_jump_threshold,
        skip_existing,
    ) = args_tuple

    result = generate_prefix_segments_for_episode(
        task_name,
        ep_idx,
        camera_name,
        task_dir,
        num_segments,
        search_window,
        target_jump_threshold,
        skip_existing,
    )

    # Save metadata if successful
    if result["success"] and result["segments"]:
        try:
            save_segment_metadata(task_dir, ep_idx, num_segments, result["segments"])
        except Exception as e:
            result["error"] = f"Failed to save metadata: {e}"
            result["success"] = False

    return result


def get_episode_count(camera_dir: Path) -> int:
    """Get the number of episodes from affordance_summary_masked.jsonl."""
    summary_path = camera_dir / "affordance_summary_masked.jsonl"
    summaries = load_jsonl(summary_path)
    return len(summaries)


def main():
    args = parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        logger.error(f"Outputs directory not found: {outputs_dir}")
        return

    # Determine which tasks to process
    if args.task:
        task_dirs = [outputs_dir / args.task]
    else:
        task_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])

    total_episodes = 0
    successful_episodes = 0
    failed_episodes = 0
    skipped_episodes = 0

    # Collect work items
    work_items = []
    for task_dir in task_dirs:
        task_name = task_dir.name

        # Find camera directories
        camera_dir = task_dir / args.camera_name
        if not camera_dir.exists():
            logger.warning(f"Camera directory not found: {camera_dir}")
            continue

        # Get episode count
        num_episodes = get_episode_count(camera_dir)
        if num_episodes == 0:
            logger.warning(f"No episodes found in {task_dir}")
            continue

        logger.info(f"Processing task {task_name} with {num_episodes} episodes")

        # Add work items for each episode
        for ep_idx in range(num_episodes):
            work_items.append(
                (
                    task_name,
                    ep_idx,
                    args.camera_name,
                    task_dir,
                    args.num_segments,
                    args.search_window,
                    args.target_jump_threshold,
                    args.skip_existing,
                )
            )

    logger.info(f"Total work items: {len(work_items)}")

    # Process episodes in parallel
    results_summary = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_episode, item) for item in work_items]

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()

                if result["success"]:
                    results_summary["success"] += 1
                    if args.verbose:
                        logger.info(
                            f"[{i}/{len(futures)}] ✓ {result['task_name']} ep{result['episode_index']:06d} "
                            f"→ {len(result['segments'])} segments"
                        )
                else:
                    if "Already segmented" in (result.get("error") or ""):
                        results_summary["skipped"] += 1
                        if args.verbose:
                            logger.info(f"[{i}/{len(futures)}] ⊘ {result['task_name']} ep{result['episode_index']:06d} (skipped)")
                    else:
                        results_summary["failed"] += 1
                        results_summary["errors"].append(
                            f"{result['task_name']} ep{result['episode_index']:06d}: {result['error']}"
                        )
                        if args.verbose:
                            logger.warning(
                                f"[{i}/{len(futures)}] ✗ {result['task_name']} ep{result['episode_index']:06d}: {result['error']}"
                            )

                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{len(futures)} | Success: {results_summary['success']} | "
                        f"Failed: {results_summary['failed']} | Skipped: {results_summary['skipped']}"
                    )

            except Exception as e:
                logger.error(f"Worker error: {e}")
                results_summary["failed"] += 1
                results_summary["errors"].append(f"Worker exception: {e}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed:  {results_summary['success'] + results_summary['failed'] + results_summary['skipped']}")
    logger.info(f"Successful:       {results_summary['success']}")
    logger.info(f"Failed:           {results_summary['failed']}")
    logger.info(f"Skipped:          {results_summary['skipped']}")

    if results_summary["errors"] and not args.verbose:
        logger.info("\nFirst 5 errors:")
        for error in results_summary["errors"][:5]:
            logger.info(f"  - {error}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
