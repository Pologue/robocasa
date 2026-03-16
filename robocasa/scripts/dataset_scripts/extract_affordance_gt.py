import argparse
import glob
import json
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import robosuite

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.scripts.dataset_scripts.playback_utils import resolve_instruction_from_ep_meta


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract affordance ground truth from RoboCasa demos by replaying simulator states. "
            "Outputs subtask boundaries and optional per-frame affordance targets."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to LeRobot-format RoboCasa dataset root (folder containing extras/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Only process first n episodes.",
    )
    parser.add_argument(
        "--camera_name",
        type=str,
        default="robot0_agentview_center",
        help="Camera used to project 3D affordance points to 2D pixels.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=512,
        help="Reference image height used for projection.",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=768,
        help="Reference image width used for projection.",
    )
    parser.add_argument(
        "--save_dense",
        action="store_true",
        help="If set, save per-frame dense records to --dense_dir (recommended with --dense_format npz/parquet).",
    )
    parser.add_argument(
        "--dense_dir",
        type=str,
        default=None,
        help="Directory for dense per-episode outputs. Required when --save_dense is set.",
    )
    parser.add_argument(
        "--dense_format",
        type=str,
        default="npz",
        choices=["jsonl", "npz", "parquet"],
        help="Storage format for dense per-frame outputs.",
    )
    parser.add_argument(
        "--segment_min_frames",
        type=int,
        default=12,
        help="Minimum frames for a segment after temporal smoothing.",
    )
    parser.add_argument(
        "--target_switch_min_frames",
        type=int,
        default=6,
        help="A focus switch must persist this many frames to become a segment boundary.",
    )
    parser.add_argument(
        "--target_jump_threshold",
        type=float,
        default=0.10,
        help="Meters. Extra split candidates are added on large target 3D jumps.",
    )
    parser.add_argument(
        "--enforce_expected_subtasks",
        action="store_true",
        help="Use docs/composite_tasks/composite_task_attributes.js hints to add missing boundaries.",
    )
    parser.add_argument(
        "--debug_video_dir",
        type=str,
        default=None,
        help="If set, writes replay videos with projected affordance markers.",
    )
    parser.add_argument(
        "--debug_video_max_episodes",
        type=int,
        default=10,
        help="Maximum number of episodes to write debug videos for.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def _safe_int(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        return int(np.array(x).reshape(-1)[0])
    try:
        return int(x)
    except Exception:
        return None


def load_task_index_to_name(dataset_root):
    """Load LeRobot task map from tasks.jsonl or tasks.json."""
    task_map = {}
    meta_dir = dataset_root / "meta"

    jsonl_path = meta_dir / "tasks.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = _safe_int(row.get("task_index"))
                name = row.get("task")
                if idx is not None and isinstance(name, str):
                    task_map[idx] = name
        if len(task_map) > 0:
            return task_map

    json_path = meta_dir / "tasks.json"
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        if isinstance(rows, dict):
            rows = [rows]
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = _safe_int(row.get("task_index"))
            name = row.get("task")
            if idx is not None and isinstance(name, str):
                task_map[idx] = name
    return task_map


def get_episode_task_name(dataset_root, ep_idx, task_map):
    """Get task name from per-episode parquet task index + meta/tasks map."""
    pattern = dataset_root / "data" / "*" / f"episode_{ep_idx:06d}.parquet"
    files = sorted(glob.glob(str(pattern)))
    if len(files) == 0:
        return ""

    df = pd.read_parquet(files[0])
    if len(df) == 0:
        return ""

    col_candidates = ["annotation.human.task_name", "task_index"]
    for col in col_candidates:
        if col in df.columns:
            idx = _safe_int(df.iloc[0][col])
            if idx is not None:
                return task_map.get(idx, "")
    return ""


def load_expected_subtasks_hint(repo_root):
    """Load task->subtasks hints from docs JS file if available."""
    hints = {}
    js_path = repo_root / "docs" / "composite_tasks" / "composite_task_attributes.js"
    if not js_path.exists():
        return hints
    text = js_path.read_text(encoding="utf-8")

    # Parse entries like: "DeliverStraw":{"mobile":"Yes","subtasks":4}
    # Minimal parser to avoid JS runtime dependencies.
    cursor = 0
    while True:
        key_start = text.find('"', cursor)
        if key_start < 0:
            break
        key_end = text.find('"', key_start + 1)
        if key_end < 0:
            break
        key = text[key_start + 1 : key_end]

        subtasks_token = '"subtasks":'
        st_pos = text.find(subtasks_token, key_end)
        if st_pos < 0:
            cursor = key_end + 1
            continue

        # Keep parser local to nearby object chunk.
        obj_close = text.find("}", key_end)
        if obj_close < 0 or st_pos > obj_close:
            cursor = key_end + 1
            continue

        num_start = st_pos + len(subtasks_token)
        num_end = num_start
        while num_end < len(text) and text[num_end].isdigit():
            num_end += 1

        if num_end > num_start:
            hints[key] = int(text[num_start:num_end])

        cursor = key_end + 1

    return hints


def world_to_pixel(env, world_xyz, camera_name, image_h, image_w):
    """Project world coordinates to image pixel with MuJoCo camera extrinsics/intrinsics."""
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
    except Exception:
        return None

    cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
    cam_rot = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)

    rel = np.array(world_xyz) - cam_pos
    cam_xyz = cam_rot.T @ rel

    # MuJoCo camera convention: looks along -Z in camera coordinates.
    z = -cam_xyz[2]
    if z <= 1e-8:
        return None

    fovy_deg = float(env.sim.model.cam_fovy[cam_id])
    fovy = np.deg2rad(fovy_deg)
    fy = 0.5 * image_h / np.tan(0.5 * fovy)
    fx = fy
    cx = 0.5 * (image_w - 1)
    cy = 0.5 * (image_h - 1)

    u = fx * (cam_xyz[0] / z) + cx
    v = -fy * (cam_xyz[1] / z) + cy

    return [float(u), float(v)]


def safe_bool(fn):
    try:
        return bool(fn())
    except Exception:
        return False


def get_gripper_pos(env):
    return np.array(env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]])


def get_fixture_center(fixture):
    if hasattr(fixture, "get_ext_sites"):
        try:
            p0, px, py, pz = fixture.get_ext_sites(relative=False)
            return np.mean(np.stack([p0, px, py, pz], axis=0), axis=0)
        except Exception:
            pass
    if hasattr(fixture, "pos"):
        try:
            return np.array(fixture.pos)
        except Exception:
            return None
    return None


def choose_target_focus(env, prev_focus_name=None):
    """
    Choose current affordance target by interaction focus:
    - Prefer nearest object to gripper.
    - Fall back to nearby fixture when object is not close.
    """
    g = get_gripper_pos(env)

    best_obj_name = None
    best_obj_xyz = None
    best_obj_dist = None
    for obj_name, body_id in env.obj_body_id.items():
        p = np.array(env.sim.data.body_xpos[body_id])
        d = float(np.linalg.norm(p - g))
        if best_obj_dist is None or d < best_obj_dist:
            best_obj_dist = d
            best_obj_name = obj_name
            best_obj_xyz = p

    best_fx_name = None
    best_fx_xyz = None
    best_fx_dist = None
    for fx_name, fixture in env.fixtures.items():
        p = get_fixture_center(fixture)
        if p is None:
            continue
        d = float(np.linalg.norm(p - g))
        if best_fx_dist is None or d < best_fx_dist:
            best_fx_dist = d
            best_fx_name = fx_name
            best_fx_xyz = p

    # If gripper is close to an object, object is likely the operative target.
    if best_obj_name is not None and best_obj_dist is not None and best_obj_dist <= 0.22:
        return best_obj_name, best_obj_xyz, "object"

    # Otherwise allow fixture interaction focus (doors, drawers, buttons, etc.).
    if best_fx_name is not None and best_fx_dist is not None and best_fx_dist <= 0.30:
        return best_fx_name, best_fx_xyz, "fixture"

    # Final fallback: nearest object.
    if best_obj_name is not None:
        return best_obj_name, best_obj_xyz, "object"

    return prev_focus_name or "", None, "unknown"


def stabilize_focus_sequence(names, min_persist):
    """Temporal smoothing: drop short-lived switches."""
    if len(names) == 0:
        return names

    out = list(names)
    i = 0
    while i < len(out):
        j = i + 1
        while j < len(out) and out[j] == out[i]:
            j += 1
        run_len = j - i
        if run_len < min_persist:
            prev_name = out[i - 1] if i > 0 else None
            next_name = out[j] if j < len(out) else None
            fill_name = prev_name if prev_name is not None else next_name
            if fill_name is not None:
                for k in range(i, j):
                    out[k] = fill_name
        i = j
    return out


def build_segments_from_focus(focus_names, min_frames):
    if len(focus_names) == 0:
        return []

    segments = []
    start = 0
    for i in range(1, len(focus_names)):
        if focus_names[i] != focus_names[i - 1]:
            segments.append({"start": start, "end": i - 1, "target": focus_names[i - 1]})
            start = i
    segments.append({"start": start, "end": len(focus_names) - 1, "target": focus_names[-1]})

    if min_frames <= 1 or len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        prev_len = prev["end"] - prev["start"] + 1
        cur_len = seg["end"] - seg["start"] + 1
        if prev_len < min_frames:
            prev["end"] = seg["end"]
        elif cur_len < min_frames:
            prev["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def add_boundaries_by_target_jump(segments, target_xyz, jump_threshold, expected_segments):
    """Add missing boundaries using large 3D target displacement peaks."""
    if expected_segments is None or len(segments) >= expected_segments:
        return segments

    if len(target_xyz) < 3:
        return segments

    deltas = []
    for i in range(1, len(target_xyz)):
        p0 = target_xyz[i - 1]
        p1 = target_xyz[i]
        if p0 is None or p1 is None:
            d = 0.0
        else:
            d = float(np.linalg.norm(np.array(p1) - np.array(p0)))
        deltas.append((i, d))

    existing = set(seg["end"] for seg in segments[:-1])
    candidates = [x for x in deltas if x[1] >= jump_threshold and x[0] not in existing]
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    needed = expected_segments - len(segments)
    if needed <= 0:
        return segments

    split_frames = sorted(x[0] for x in candidates[:needed])

    # If jump-based candidates are insufficient, backfill by uniform temporal splits.
    if len(split_frames) < needed:
        n = len(target_xyz)
        if n > 1:
            for k in range(1, expected_segments):
                sf = int(round((k * n) / expected_segments))
                if sf <= 0 or sf >= n:
                    continue
                if sf in existing or sf in split_frames:
                    continue
                split_frames.append(sf)
                if len(split_frames) >= needed:
                    break

    split_frames = sorted(split_frames)
    if len(split_frames) == 0:
        return segments

    new_segments = []
    prev = 0
    for sf in split_frames:
        if sf <= prev:
            continue
        new_segments.append({"start": prev, "end": sf - 1})
        prev = sf
    new_segments.append({"start": prev, "end": len(target_xyz) - 1})

    for seg in new_segments:
        st, ed = seg["start"], seg["end"]
        names = [s["target"] for s in segments if not (s["end"] < st or s["start"] > ed)]
        seg["target"] = names[0] if len(names) > 0 else ""

    return new_segments


def draw_cross(img, uv, color=(255, 32, 32), size=6):
    if uv is None:
        return img
    h, w = img.shape[:2]
    u = int(round(uv[0]))
    v = int(round(uv[1]))
    if u < 0 or u >= w or v < 0 or v >= h:
        return img

    out = img.copy()
    for d in range(-size, size + 1):
        x = u + d
        y = v
        if 0 <= x < w:
            out[y, x] = color
    for d in range(-size, size + 1):
        x = u
        y = v + d
        if 0 <= y < h:
            out[y, x] = color
    return out


def write_dense_episode(dense_dir, dense_format, ep_idx, dense_payload):
    dense_dir.mkdir(parents=True, exist_ok=True)
    if dense_format == "jsonl":
        path = dense_dir / f"episode_{ep_idx:06d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i in range(len(dense_payload["frame"])):
                row = {
                    "frame": int(dense_payload["frame"][i]),
                    "target_name": dense_payload["target_name"][i],
                    "target_type": dense_payload["target_type"][i],
                    "target_xyz": dense_payload["target_xyz"][i],
                    "target_uv": dense_payload["target_uv"][i],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)

    if dense_format == "npz":
        path = dense_dir / f"episode_{ep_idx:06d}.npz"
        xyz = np.array(
            [[np.nan, np.nan, np.nan] if x is None else x for x in dense_payload["target_xyz"]],
            dtype=np.float32,
        )
        uv = np.array(
            [[np.nan, np.nan] if x is None else x for x in dense_payload["target_uv"]],
            dtype=np.float32,
        )
        np.savez_compressed(
            path,
            frame=np.array(dense_payload["frame"], dtype=np.int32),
            target_name=np.array(dense_payload["target_name"], dtype=object),
            target_type=np.array(dense_payload["target_type"], dtype=object),
            target_xyz=xyz,
            target_uv=uv,
        )
        return str(path)

    path = dense_dir / f"episode_{ep_idx:06d}.parquet"
    df = pd.DataFrame(
        {
            "frame": dense_payload["frame"],
            "target_name": dense_payload["target_name"],
            "target_type": dense_payload["target_type"],
            "target_xyz": dense_payload["target_xyz"],
            "target_uv": dense_payload["target_uv"],
        }
    )
    df.to_parquet(path, index=False)
    return str(path)


def build_env(dataset_root, need_offscreen=False):
    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = bool(need_offscreen)
    # Keep camera observations disabled to avoid dataset-meta camera key mismatches
    # (e.g. 'agentview') during observable initialization. We render explicitly via sim.render.
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def episode_record(
    env,
    dataset_root,
    ep_idx,
    camera_name,
    image_h,
    image_w,
    save_dense,
    dense_dir,
    dense_format,
    segment_min_frames,
    target_switch_min_frames,
    target_jump_threshold,
    expected_subtasks,
    write_debug_video,
    debug_video_dir,
):
    states = LU.get_episode_states(dataset_root, ep_idx)
    ep_meta = LU.get_episode_meta(dataset_root, ep_idx)
    model_xml = LU.get_episode_model_xml(dataset_root, ep_idx)

    initial_state = {
        "states": states[0],
        "model": model_xml,
        "ep_meta": json.dumps(ep_meta),
    }
    reset_to(env, initial_state)

    instruction = resolve_instruction_from_ep_meta(ep_meta)

    focus_names = []
    focus_types = []
    target_xyz = []
    target_uv = []

    video_writer = None
    if write_debug_video:
        debug_video_dir.mkdir(parents=True, exist_ok=True)
        video_path = debug_video_dir / f"episode_{ep_idx:06d}.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=20)

    prev_focus_name = None

    for t in range(states.shape[0]):
        reset_to(env, {"states": states[t]})

        focus_name, xyz, focus_type = choose_target_focus(env, prev_focus_name)
        prev_focus_name = focus_name

        uv = None
        if xyz is not None:
            uv = world_to_pixel(
                env,
                xyz,
                camera_name=camera_name,
                image_h=image_h,
                image_w=image_w,
            )

        xyz_out = None if xyz is None else [float(v) for v in xyz]

        focus_names.append(focus_name)
        focus_types.append(focus_type)
        target_xyz.append(xyz_out)
        target_uv.append(uv)

        if video_writer is not None:
            frame = env.sim.render(
                height=image_h,
                width=image_w,
                camera_name=camera_name,
            )[::-1]
            frame = draw_cross(frame, uv)
            video_writer.append_data(frame)

    if video_writer is not None:
        video_writer.close()

    smoothed_focus = stabilize_focus_sequence(focus_names, min_persist=target_switch_min_frames)
    segments = build_segments_from_focus(smoothed_focus, min_frames=segment_min_frames)
    if expected_subtasks is not None and expected_subtasks > 0:
        segments = add_boundaries_by_target_jump(
            segments,
            target_xyz,
            jump_threshold=target_jump_threshold,
            expected_segments=expected_subtasks,
        )

    boundaries = []
    for seg_id, seg in enumerate(segments):
        boundaries.append(
            {
                "segment_id": int(seg_id),
                "start_frame": int(seg["start"]),
                "end_frame": int(seg["end"]),
                "target_name": seg.get("target", ""),
            }
        )

    dense_path = None
    if save_dense:
        dense_payload = {
            "frame": list(range(states.shape[0])),
            "target_name": smoothed_focus,
            "target_type": focus_types,
            "target_xyz": target_xyz,
            "target_uv": target_uv,
        }
        dense_path = write_dense_episode(dense_dir, dense_format, ep_idx, dense_payload)

    return {
        "episode_index": int(ep_idx),
        "task": "",
        "instruction": instruction,
        "num_frames": int(states.shape[0]),
        "camera_name": camera_name,
        "image_size": [int(image_h), int(image_w)],
        "subtask_boundaries": boundaries,
        "num_segments": len(boundaries),
        "expected_num_subtasks": expected_subtasks,
        "dense_path": dense_path,
    }


def main():
    args = parse_args()

    dataset_root = Path(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_dense and args.dense_dir is None:
        raise ValueError("--save_dense requires --dense_dir")

    dense_dir = Path(args.dense_dir) if args.dense_dir is not None else None
    debug_video_dir = (
        Path(args.debug_video_dir) if args.debug_video_dir is not None else None
    )

    task_map = load_task_index_to_name(dataset_root)
    repo_root = Path(__file__).resolve().parents[3]
    subtasks_hint = load_expected_subtasks_hint(repo_root)

    episodes = LU.get_episodes(dataset_root)
    if args.n is not None:
        episodes = episodes[: args.n]

    need_video = args.debug_video_dir is not None
    env = build_env(dataset_root, need_offscreen=need_video)

    with out_path.open("w", encoding="utf-8") as f:
        for i, _ in enumerate(episodes):
            if args.verbose:
                print(f"[episode {i}] extracting...")

            task_name = get_episode_task_name(dataset_root, i, task_map)
            expected_subtasks = None
            if args.enforce_expected_subtasks and task_name in subtasks_hint:
                expected_subtasks = int(subtasks_hint[task_name])

            write_debug_video = (
                debug_video_dir is not None and i < int(args.debug_video_max_episodes)
            )

            record = episode_record(
                env=env,
                dataset_root=dataset_root,
                ep_idx=i,
                camera_name=args.camera_name,
                image_h=args.camera_height,
                image_w=args.camera_width,
                save_dense=args.save_dense,
                dense_dir=dense_dir,
                dense_format=args.dense_format,
                segment_min_frames=args.segment_min_frames,
                target_switch_min_frames=args.target_switch_min_frames,
                target_jump_threshold=args.target_jump_threshold,
                expected_subtasks=expected_subtasks,
                write_debug_video=write_debug_video,
                debug_video_dir=debug_video_dir,
            )
            record["task"] = task_name
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.verbose:
                num_b = len(record["subtask_boundaries"])
                print(
                    f"[episode {i}] done, task={task_name}, segments={num_b}, expected={record['expected_num_subtasks']}"
                )

    env.close()
    print(f"Saved affordance annotations to: {out_path}")


if __name__ == "__main__":
    main()
