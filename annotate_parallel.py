#!/usr/bin/env python3
"""
Parallel affordance annotation for RoboCasa composite tasks.
Processes multiple tasks concurrently across multiple GPUs.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple


# Camera names to use for each task
CAMERAS = [
    "robot0_agentview_center",
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_frontview",
    "robot0_eye_in_hand",
]


def find_composite_tasks(composite_root: Path, max_tasks: int = None) -> List[Path]:
    """
    Discover all composite task datasets in order.
    
    Args:
        composite_root: Path to datasets/v1.0/pretrain/composite
        max_tasks: Maximum number of tasks to process (None for all)
    
    Returns:
        List of paths to task lerobot directories
    """
    if not composite_root.exists():
        raise FileNotFoundError(f"Composite root not found: {composite_root}")
    
    task_dirs = []
    
    # Iterate over task folders
    for task_folder in sorted(composite_root.iterdir()):
        if not task_folder.is_dir():
            continue
        
        # Find date subfolders
        for date_folder in sorted(task_folder.iterdir()):
            if not date_folder.is_dir():
                continue
            
            lerobot_path = date_folder / "lerobot"
            if lerobot_path.exists():
                task_dirs.append(lerobot_path)
                
                if max_tasks and len(task_dirs) >= max_tasks:
                    return task_dirs
    
    return task_dirs


def run_annotation_for_camera(
    dataset_path: str,
    task_name: str,
    camera_name: str,
    gpu_id: int,
    num_episodes: int = 1,
    segment_min_frames: int = 12,
    target_switch_min_frames: int = 6,
    target_jump_threshold: float = 0.10,
    min_visible_pixels: int = 24,
    enforce_expected_subtasks: bool = True,
    debug_video: bool = True,
    verbose: bool = False,
) -> Tuple[str, str, bool, str]:
    """
    Run affordance extraction for a single camera on a task.
    
    Args:
        dataset_path: Path to the lerobot dataset
        task_name: Task name for output organization
        camera_name: Camera to use for annotation
        gpu_id: GPU device ID to use
        num_episodes: Number of episodes to process
        Other args: Extraction parameters
    
    Returns:
        Tuple of (task_name, camera_name, success, message)
    """
    try:
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return task_name, camera_name, False, f"Dataset not found: {dataset_path}"
        
        output_dir = Path(f"outputs/{task_name}/{camera_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = output_dir / "affordance_summary_masked.jsonl"
        dense_dir = output_dir / "dense_masked"
        debug_video_dir = output_dir / "debug_videos_masked" if debug_video else None
        
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "robocasa.scripts.dataset_scripts.extract_affordance_gt",
            "--dataset", str(dataset_path),
            "--output", str(summary_path),
            "--n", str(num_episodes),
            "--camera_name", camera_name,
            "--save_dense",
            "--dense_dir", str(dense_dir),
            "--dense_format", "npz",
            "--segment_min_frames", str(segment_min_frames),
            "--target_switch_min_frames", str(target_switch_min_frames),
            "--target_jump_threshold", str(target_jump_threshold),
            "--min_visible_pixels", str(min_visible_pixels),
        ]
        
        if debug_video:
            cmd.extend([
                "--debug_video_dir", str(debug_video_dir),
                "--debug_video_max_episodes", str(num_episodes),
            ])
        
        if enforce_expected_subtasks:
            cmd.append("--enforce_expected_subtasks")
        
        if verbose:
            cmd.append("--verbose")
        
        # Set GPU environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Run extraction
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout
            return task_name, camera_name, False, f"Command failed: {error_msg[:200]}"
        
        return task_name, camera_name, True, f"Saved to {summary_path}"
    
    except subprocess.TimeoutExpired:
        return task_name, camera_name, False, "Timeout (>1 hour)"
    except Exception as e:
        return task_name, camera_name, False, f"Error: {str(e)[:200]}"


def main():
    parser = argparse.ArgumentParser(
        description="Parallel affordance annotation for RoboCasa composite tasks."
    )
    parser.add_argument(
        "--composite_root",
        type=str,
        default="datasets/v1.0/pretrain/composite",
        help="Path to composite tasks root directory.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to process (None for all).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum number of parallel processes.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to annotate per task.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs available (GPUs will be cycled).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=CAMERAS,
        help="List of camera names to use.",
    )
    parser.add_argument(
        "--segment_min_frames",
        type=int,
        default=12,
        help="Minimum frames for a segment.",
    )
    parser.add_argument(
        "--target_switch_min_frames",
        type=int,
        default=6,
        help="Frames for focus switch to persist.",
    )
    parser.add_argument(
        "--target_jump_threshold",
        type=float,
        default=0.10,
        help="Threshold for target 3D jump.",
    )
    parser.add_argument(
        "--min_visible_pixels",
        type=int,
        default=24,
        help="Minimum visible pixels for affordance annotation.",
    )
    parser.add_argument(
        "--no_expected_subtasks",
        action="store_true",
        help="Do not enforce expected subtask count.",
    )
    parser.add_argument(
        "--no_debug_video",
        action="store_true",
        help="Do not save debug videos.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    
    args = parser.parse_args()
    
    # Discover tasks
    composite_root = Path(args.composite_root)
    task_datasets = find_composite_tasks(composite_root, max_tasks=args.max_tasks)
    
    if not task_datasets:
        print("No composite tasks found.")
        return
    
    print(f"Found {len(task_datasets)} task(s) to process.")
    
    # Build work items: (dataset_path, task_name, camera_name, gpu_id)
    work_items = []
    gpu_id_cycle = 0
    
    for dataset_path in task_datasets:
        task_name = dataset_path.parent.parent.name
        
        for camera in args.cameras:
            work_items.append({
                "dataset_path": str(dataset_path),
                "task_name": task_name,
                "camera_name": camera,
                "gpu_id": gpu_id_cycle % args.num_gpus,
                "num_episodes": args.num_episodes,
                "segment_min_frames": args.segment_min_frames,
                "target_switch_min_frames": args.target_switch_min_frames,
                "target_jump_threshold": args.target_jump_threshold,
                "min_visible_pixels": args.min_visible_pixels,
                "enforce_expected_subtasks": not args.no_expected_subtasks,
                "debug_video": not args.no_debug_video,
                "verbose": args.verbose,
            })
            gpu_id_cycle += 1
    
    print(f"Total work items: {len(work_items)} (tasks × cameras)")
    
    # Process with thread pool
    results = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_annotation_for_camera, **item): item["task_name"]
            for item in work_items
        }
        
        for i, future in enumerate(as_completed(futures)):
            task_name, camera_name, success, message = future.result()
            status = "✓" if success else "✗"
            print(f"[{i+1}/{len(work_items)}] {status} {task_name:30s} {camera_name:30s} {message}")
            results.append((task_name, camera_name, success, message))
    
    # Summary
    print("\n" + "=" * 80)
    successful = sum(1 for _, _, s, _ in results if s)
    print(f"Completed: {successful}/{len(results)} items successfully")
    print("Output directory: outputs/")


if __name__ == "__main__":
    main()
