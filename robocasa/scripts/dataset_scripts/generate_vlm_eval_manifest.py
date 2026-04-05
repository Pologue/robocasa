"""
Complete workflow for generating and evaluating prefix video segments.

This module combines:
1. Segment generation from affordance annotations
2. Prefix video file generation
3. VLM evaluation manifest preparation

Usage:
    # Step 1: Generate segment metadata
    python generate_prefix_segments.py --outputs-dir outputs --num-segments 4 --num-workers 8
    
    # Step 2: Generate video files
    python generate_prefix_videos.py --dataset /path/to/lerobot/dataset --outputs-dir outputs --num-segments 4 --num-workers 8
    
    # Step 3: Prepare VLM evaluation manifest (optional)
    python generate_vlm_eval_manifest.py --outputs-dir outputs --num-segments 4 --output-manifest vlm_eval_manifest.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
    return rows


def get_segment_dirs(task_dir: Path, num_segments: int) -> List[Path]:
    """Get segment directories, preferring prefix_videos and keeping legacy compatibility."""
    prefix_root = task_dir / "prefix_videos"
    if prefix_root.exists():
        return sorted(prefix_root.glob(f"episode_*_prefix_{num_segments}seg"))
    return sorted(task_dir.glob(f"episode_*_prefix_{num_segments}seg"))


def generate_vlm_eval_manifest(
    outputs_dir: Path,
    num_segments: int,
    camera_name: str = "robot0_agentview_center",
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Generate VLM evaluation manifest from segmented datasets.
    
    Creates a JSONL manifest with:
    - Video path
    - Task and instruction
    - Segment boundaries
    - Ground truth affordance info
    
    Useful for feeding into VLM affordance prediction pipelines.
    """
    manifest_entries = []

    task_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])

    for task_dir in task_dirs:
        task_name = task_dir.name

        # Find segment directories
        for segment_dir in get_segment_dirs(task_dir, num_segments):
            metadata_path = segment_dir / "segments_metadata.json"
            if not metadata_path.exists():
                continue

            metadata = load_json(metadata_path)
            if not metadata or "segments" not in metadata:
                continue

            ep_idx = metadata["episode_index"]

            # Try to load original affordance annotation
            camera_dir = task_dir / camera_name
            summary_path = camera_dir / "affordance_summary_masked.jsonl"
            original_annotation = None

            if summary_path.exists():
                summaries = load_jsonl(summary_path)
                for s in summaries:
                    if s.get("episode_index") == ep_idx:
                        original_annotation = s
                        break

            # Create manifest entry for each segment
            for segment in metadata["segments"]:
                seg_idx = segment["segment_index"]
                video_path = segment_dir / "videos" / f"segment_{seg_idx:02d}.mp4"

                if not video_path.exists():
                    continue

                entry = {
                    "task": task_name,
                    "episode_index": ep_idx,
                    "segment_index": seg_idx,
                    "video_path": str(video_path),
                    "segment_start_frame": segment["start_frame"],
                    "segment_end_frame": segment["end_frame"],
                    "target_name": segment.get("target_name", ""),
                    "target_visible_at_segment_end": segment.get("target_visible_at_end", False),
                    "target_visible_pixels_at_segment_end": segment.get("target_visible_pixels_at_end", 0),
                    "instruction": original_annotation.get("instruction", "") if original_annotation else "",
                    "num_total_segments": len(metadata["segments"]),
                }

                manifest_entries.append(entry)

    # Save manifest if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return manifest_entries


def generate_stats_summary(
    outputs_dir: Path,
    num_segments: int,
    camera_name: str = "robot0_agentview_center",
) -> Dict:
    """Generate statistics summary of the segmented dataset."""
    stats = {
        "total_tasks": 0,
        "total_episodes": 0,
        "total_segments": 0,
        "tasks_with_videos": 0,
        "segments_with_videos": 0,
        "target_visibility_distribution": {},
    }

    task_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()])

    for task_dir in task_dirs:
        task_episodes = 0
        task_segments = 0
        task_segments_with_videos = 0

        # Find segment directories
        for segment_dir in get_segment_dirs(task_dir, num_segments):
            metadata_path = segment_dir / "segments_metadata.json"
            if not metadata_path.exists():
                continue

            metadata = load_json(metadata_path)
            if not metadata or "segments" not in metadata:
                continue

            task_episodes += 1
            stats["total_episodes"] += 1

            # Check each segment
            for segment in metadata["segments"]:
                task_segments += 1
                stats["total_segments"] += 1

                video_path = segment_dir / "videos" / f"segment_{segment['segment_index']:02d}.mp4"
                if video_path.exists():
                    task_segments_with_videos += 1
                    stats["segments_with_videos"] += 1

                # Track visibility distribution
                visible = segment.get("target_visible_at_end", False)
                key = "visible" if visible else "invisible"
                stats["target_visibility_distribution"][key] = (
                    stats["target_visibility_distribution"].get(key, 0) + 1
                )

        if task_episodes > 0:
            stats["tasks_with_videos"] += 1
        stats["total_tasks"] += 1

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLM evaluation manifest generation utility."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Manifest generation
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Generate VLM evaluation manifest",
    )
    manifest_parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Root outputs directory.",
    )
    manifest_parser.add_argument(
        "--num-segments",
        type=int,
        required=True,
        help="Number of segments.",
    )
    manifest_parser.add_argument(
        "--output",
        type=str,
        default="vlm_eval_manifest.jsonl",
        help="Output manifest file path.",
    )
    manifest_parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_agentview_center",
        help="Camera name.",
    )

    # Statistics
    stats_parser = subparsers.add_parser(
        "stats",
        help="Generate dataset statistics",
    )
    stats_parser.add_argument(
        "--outputs-dir",
        type=str,
        required=True,
        help="Root outputs directory.",
    )
    stats_parser.add_argument(
        "--num-segments",
        type=int,
        required=True,
        help="Number of segments.",
    )
    stats_parser.add_argument(
        "--camera-name",
        type=str,
        default="robot0_agentview_center",
        help="Camera name.",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()

    if args.command == "manifest":
        outputs_dir = Path(args.outputs_dir)
        if not outputs_dir.exists():
            logger.error(f"Outputs directory not found: {outputs_dir}")
            return

        logger.info("Generating VLM evaluation manifest...")
        manifest = generate_vlm_eval_manifest(
            outputs_dir,
            args.num_segments,
            args.camera_name,
            Path(args.output),
        )
        logger.info(f"Generated {len(manifest)} manifest entries")
        logger.info(f"Manifest saved to: {args.output}")

    elif args.command == "stats":
        outputs_dir = Path(args.outputs_dir)
        if not outputs_dir.exists():
            logger.error(f"Outputs directory not found: {outputs_dir}")
            return

        logger.info("Computing dataset statistics...")
        stats = generate_stats_summary(
            outputs_dir,
            args.num_segments,
            args.camera_name,
        )

        logger.info("\n" + "=" * 60)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total tasks:                  {stats['total_tasks']}")
        logger.info(f"Total episodes:               {stats['total_episodes']}")
        logger.info(f"Total segments:               {stats['total_segments']}")
        logger.info(f"Tasks with videos:            {stats['tasks_with_videos']}")
        logger.info(f"Segments with videos:         {stats['segments_with_videos']}")

        visible = stats["target_visibility_distribution"].get("visible", 0)
        invisible = stats["target_visibility_distribution"].get("invisible", 0)
        total_vis = visible + invisible
        if total_vis > 0:
            logger.info("\nTarget Visibility:")
            logger.info(f"  Visible:                    {visible} ({100.0*visible/total_vis:.1f}%)")
            logger.info(f"  Invisible:                  {invisible} ({100.0*invisible/total_vis:.1f}%)")
        logger.info("=" * 60)

    else:
        parser = parse_args()
        parser.print_help()


if __name__ == "__main__":
    main()
