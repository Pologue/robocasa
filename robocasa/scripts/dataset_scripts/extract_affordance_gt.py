import argparse
import inspect
import json
import re
from pathlib import Path

import numpy as np
import robosuite

import robocasa  # noqa: F401
import robocasa.utils.lerobot_utils as LU
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.scripts.dataset_scripts.playback_utils import resolve_instruction_from_ep_meta


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract affordance ground truth from RoboCasa demos by replaying simulator states. "
            "Outputs subtask boundaries (predicate transitions) and optional per-frame targets."
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
        help="If set, include per-frame records in output (large files).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def discover_predicates(env):
    """Return predicate callables in a stable order for stage decomposition."""
    method_names = []
    for name in dir(env):
        if not name.startswith("_check_success"):
            continue
        if name == "_check_success":
            continue
        fn = getattr(env, name, None)
        if callable(fn):
            method_names.append(name)

    method_names = sorted(method_names)

    predicates = []
    for name in method_names:
        predicates.append((name, getattr(env, name)))

    if len(predicates) == 0:
        predicates = [("_check_success", getattr(env, "_check_success"))]

    return predicates


def extract_symbol_refs_from_source(fn, valid_names):
    """Extract string literals that match known object / fixture names."""
    refs = []
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return refs

    literals = re.findall(r"['\"]([A-Za-z0-9_]+)['\"]", src)
    seen = set()
    for lit in literals:
        if lit in valid_names and lit not in seen:
            seen.add(lit)
            refs.append(lit)
    return refs


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


def choose_active_predicate(predicate_values):
    for name, value in predicate_values.items():
        if not value:
            return name
    return None


def mean_target_xyz(env, object_names):
    pts = []
    for name in object_names:
        if name in env.obj_body_id:
            pts.append(np.array(env.sim.data.body_xpos[env.obj_body_id[name]]))
    if len(pts) == 0:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def fallback_nearest_object(env):
    g = get_gripper_pos(env)
    best_name = None
    best_dist = None
    best_xyz = None
    for obj_name, body_id in env.obj_body_id.items():
        p = np.array(env.sim.data.body_xpos[body_id])
        d = float(np.linalg.norm(p - g))
        if best_dist is None or d < best_dist:
            best_dist = d
            best_name = obj_name
            best_xyz = p
    if best_name is None:
        return [], None
    return [best_name], best_xyz


def build_env(dataset_root, need_offscreen=False):
    env_meta = LU.get_env_metadata(dataset_root)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_renderer"] = False
    env_kwargs["renderer"] = "mjviewer"
    env_kwargs["has_offscreen_renderer"] = bool(need_offscreen)
    env_kwargs["use_camera_obs"] = False
    return robosuite.make(**env_kwargs)


def episode_record(env, dataset_root, ep_idx, camera_name, image_h, image_w, save_dense):
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

    predicates = discover_predicates(env)

    valid_refs = set(env.objects.keys()) | set(env.fixtures.keys())
    predicate_obj_refs = {}
    for pname, pfn in predicates:
        predicate_obj_refs[pname] = extract_symbol_refs_from_source(pfn, valid_refs)

    transitions = []
    prev_values = None
    dense_frames = []

    predicate_order = [name for name, _ in predicates]

    for t in range(states.shape[0]):
        reset_to(env, {"states": states[t]})

        values = {}
        for pname, pfn in predicates:
            values[pname] = safe_bool(pfn)

        if prev_values is not None:
            for pname in predicate_order:
                if (not prev_values[pname]) and values[pname]:
                    transitions.append(
                        {
                            "frame": int(t),
                            "predicate": pname,
                            "event": "predicate_became_true",
                        }
                    )

        active_predicate = choose_active_predicate(values)

        if active_predicate is not None:
            target_names = list(predicate_obj_refs.get(active_predicate, []))
            xyz = mean_target_xyz(env, target_names)
            if xyz is None:
                target_names, xyz = fallback_nearest_object(env)
        else:
            target_names, xyz = fallback_nearest_object(env)

        uv = None
        if xyz is not None:
            uv = world_to_pixel(
                env,
                xyz,
                camera_name=camera_name,
                image_h=image_h,
                image_w=image_w,
            )

        if save_dense:
            dense_frames.append(
                {
                    "frame": int(t),
                    "predicate_values": values,
                    "active_predicate": active_predicate,
                    "target_object_names": target_names,
                    "target_xyz": None if xyz is None else [float(v) for v in xyz],
                    "target_uv": uv,
                }
            )

        prev_values = values

    return {
        "episode_index": int(ep_idx),
        "task": ep_meta.get("env_name", ""),
        "instruction": instruction,
        "num_frames": int(states.shape[0]),
        "camera_name": camera_name,
        "image_size": [int(image_h), int(image_w)],
        "predicate_order": predicate_order,
        "predicate_object_refs": predicate_obj_refs,
        "subtask_boundaries": transitions,
        "frames": dense_frames if save_dense else None,
    }


def main():
    args = parse_args()

    dataset_root = Path(args.dataset)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = LU.get_episodes(dataset_root)
    if args.n is not None:
        episodes = episodes[: args.n]

    env = build_env(dataset_root, need_offscreen=False)

    with out_path.open("w", encoding="utf-8") as f:
        for i, _ in enumerate(episodes):
            if args.verbose:
                print(f"[episode {i}] extracting...")

            record = episode_record(
                env=env,
                dataset_root=dataset_root,
                ep_idx=i,
                camera_name=args.camera_name,
                image_h=args.camera_height,
                image_w=args.camera_width,
                save_dense=args.save_dense,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.verbose:
                num_b = len(record["subtask_boundaries"])
                print(f"[episode {i}] done, boundaries={num_b}")

    env.close()
    print(f"Saved affordance annotations to: {out_path}")


if __name__ == "__main__":
    main()
