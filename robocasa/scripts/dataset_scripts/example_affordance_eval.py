import argparse
import json
from pathlib import Path

import imageio
import numpy as np

import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.extract_affordance_gt import build_env
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small-scale affordance benchmark from RoboCasa annotations. "
            "The script exports prefix videos plus ground-truth affordance boxes so you can wire them to a VLM."
        )
    )
    parser.add_argument("--dataset", type=str, required=True, help="LeRobot dataset root.")
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="JSONL produced by extract_affordance_gt.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for prefix videos and manifest.jsonl.",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="robot0_agentview_center",
        help="Camera used to export prefix videos.",
    )
    parser.add_argument("--camera_height", type=int, default=512)
    parser.add_argument("--camera_width", type=int, default=768)
    parser.add_argument(
        "--prefix_ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Fractions of the episode to keep as VLM input prefixes.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=5,
        help="Only prepare a small number of episodes for feasibility testing.",
    )
    parser.add_argument(
        "--skip_invisible",
        action="store_true",
        help="Skip prefixes whose target is fully occluded at the last prefix frame.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_dense_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "frame": data["frame"],
        "target_name": data["target_name"],
        "target_bbox": data["target_bbox"],
        "target_visible": data["target_visible"],
        "target_visible_pixel_count": data["target_visible_pixel_count"],
    }


def build_prompt(task, instruction):
    return (
        "You are given the beginning of a robot manipulation video. "
        f"The overall task is: {task}. "
        f"Language instruction: {instruction}. "
        "Predict the current affordance region for the robot's next interaction. "
        "Return one bounding box in pixel coordinates as JSON with keys x0, y0, x1, y1."
    )


def bbox_is_valid(bbox):
    if bbox is None:
        return False
    bbox = np.asarray(bbox, dtype=np.float32)
    return np.all(np.isfinite(bbox)) and bbox.shape == (4,)


def bbox_iou(box_a, box_b):
    xa0, ya0, xa1, ya1 = [float(v) for v in box_a]
    xb0, yb0, xb1, yb1 = [float(v) for v in box_b]
    inter_x0 = max(xa0, xb0)
    inter_y0 = max(ya0, yb0)
    inter_x1 = min(xa1, xb1)
    inter_y1 = min(ya1, yb1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
    area_b = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
    denom = area_a + area_b - inter_area
    return 0.0 if denom <= 0.0 else inter_area / denom


def make_prefix_video(env, dataset_root, ep_idx, end_frame, video_path, camera_name, image_h, image_w):
    states = LU.get_episode_states(dataset_root, ep_idx)
    ep_meta = LU.get_episode_meta(dataset_root, ep_idx)
    model_xml = LU.get_episode_model_xml(dataset_root, ep_idx)

    reset_to(
        env,
        {
            "states": states[0],
            "model": model_xml,
            "ep_meta": json.dumps(ep_meta),
        },
    )

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(video_path), fps=20) as writer:
        for frame_idx in range(end_frame + 1):
            reset_to(env, {"states": states[frame_idx]})
            frame = env.sim.render(
                height=image_h,
                width=image_w,
                camera_name=camera_name,
            )[::-1]
            writer.append_data(frame)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_jsonl(args.annotations)
    annotations = annotations[: args.max_episodes]
    env = build_env(dataset_root, need_offscreen=True)
    manifest_path = output_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for record in annotations:
            dense_path = record.get("dense_path")
            if not dense_path:
                continue
            dense = load_dense_npz(dense_path)
            num_frames = int(record["num_frames"])

            for ratio in args.prefix_ratios:
                end_frame = max(0, min(num_frames - 1, int(round(num_frames * ratio)) - 1))
                gt_visible = bool(dense["target_visible"][end_frame])
                gt_bbox = dense["target_bbox"][end_frame].tolist()

                if args.skip_invisible and not gt_visible:
                    continue
                if not bbox_is_valid(gt_bbox):
                    continue

                video_name = f"episode_{record['episode_index']:06d}_prefix_{end_frame:04d}.mp4"
                video_path = output_dir / "videos" / video_name
                make_prefix_video(
                    env=env,
                    dataset_root=dataset_root,
                    ep_idx=int(record["episode_index"]),
                    end_frame=end_frame,
                    video_path=video_path,
                    camera_name=args.camera_name,
                    image_h=args.camera_height,
                    image_w=args.camera_width,
                )

                sample = {
                    "episode_index": int(record["episode_index"]),
                    "task": record.get("task", ""),
                    "instruction": record.get("instruction", ""),
                    "prefix_end_frame": end_frame,
                    "video_path": str(video_path),
                    "gt_target_name": str(dense["target_name"][end_frame]),
                    "gt_bbox": gt_bbox,
                    "gt_visible": gt_visible,
                    "prompt": build_prompt(record.get("task", ""), record.get("instruction", "")),
                }
                manifest_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

    env.close()
    print(f"Prepared prefix benchmark manifest at: {manifest_path}")
    print("Next step: run your VLM on each manifest row and compare predicted bbox against gt_bbox with IoU.")
    print(f"Example IoU sanity check: {bbox_iou([0, 0, 10, 10], [2, 2, 8, 8]):.3f}")


if __name__ == "__main__":
    main()