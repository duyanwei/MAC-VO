"""
MAC-VO benchmark on TSRB stereo rosbags.

Reads bag files directly — no ROS nodes, no image export needed.
Camera intrinsics are fixed (hardcoded from the TSRB RealSense D435i).

Output per sequence, per round:
  {seq}_AllFrameTrajectory.txt  — TUM format: timestamp tx ty tz qx qy qz qw
  {seq}_latency.txt             — per-frame wall-clock time: timestamp_s elapsed_s

Usage (from MAC-VO root with the venv active):

  # Full TSRB dataset
  python bench_rosbag.py --result_dir /mnt/DATA/experiments/mac_vo/

  # Single bag (quick test)
  python bench_rosbag.py \\
      --bagfile /mnt/IVALAB/rosbags/tsrb/multi_run/path1_1_ordered/path1_1_0.bag \\
      --seq path1_1_ordered \\
      --result_dir /tmp/mac_vo_test/

  # Specific sequence from the full dataset
  python bench_rosbag.py --seq path1_1_ordered --result_dir /mnt/DATA/experiments/mac_vo/
"""

import sys
import glob
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import torch
import pypose as pp
import rerun as rr

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from DataLoader.Interface import StereoData, StereoFrame
from Odometry.MACVO import MACVO
from Utility.Config import load_config, asNamespace
from Utility.Sandbox import Sandbox
from Utility.Timer import Timer
from Utility.Visualize import rr_plt
from evaluation_scripts.rosbag_stereo_stream import RosbagStereoStream


# ── Fixed camera intrinsics (TSRB RealSense D435i @ 640×480) ──────────────────

TSRB_K = torch.tensor([
    [378.7210693359375, 0.0, 324.4961853027344],
    [0.0, 378.7210693359375, 239.1063690185547],
    [0.0, 0.0, 1.0],
], dtype=torch.float32).unsqueeze(0)   # shape [1, 3, 3]

TSRB_BASELINE = 0.05  # metres
TSRB_WIDTH = 640
TSRB_HEIGHT = 480
TSRB_T_BS = pp.identity_SE3(1)  # identity: sensor frame = body frame

# ── Dataset ────────────────────────────────────────────────────────────────────

BAG_ROOT = "/mnt/IVALAB/rosbags/tsrb/multi_run"   # override with --bag_root

TSRB_SEQUENCES = [
    "path1_1_ordered",
    "path1_1_1_ordered",
    "path1_2_ordered",
    "path2_1_ordered",
    "north_path_1_disordered",
    "path1_1_disordered",
    "path1_2_disordered",
    "path1_3_disordered",
]


def find_bagfiles(seq: str, bag_root: str = BAG_ROOT) -> list[str]:
    """Return sorted bag segments for a sequence under bag_root/{seq}/*.bag."""
    pattern = str(Path(bag_root) / seq / "*.bag")
    bags = sorted(glob.glob(pattern))
    if not bags:
        raise FileNotFoundError(f"No bags for sequence '{seq}' (pattern: {pattern})")
    return bags


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_tum_trajectory(poses_7: np.ndarray, timestamps_ns: np.ndarray, path: Path) -> None:
    """
    Save trajectory in TUM format: timestamp_s tx ty tz qx qy qz qw

    poses_7 shape: [N, 7] — SE3 layout [x, y, z, qx, qy, qz, qw]
    timestamps_ns shape: [N]
    """
    timestamps_s = timestamps_ns.astype(np.float64) * 1e-9
    data = np.concatenate([timestamps_s[:, np.newaxis], poses_7], axis=1)
    np.savetxt(path, data, fmt="%.9f")


def save_latency(records: list[tuple[float, float]], path: Path) -> None:
    """
    Save per-frame latency: timestamp_s elapsed_s
    records: list of (image_timestamp_s, elapsed_seconds)
    """
    arr = np.array(records, dtype=np.float64)
    np.savetxt(path, arr, fmt="%.9f", header="image_timestamp_s elapsed_s")


def extract_trajectory(system: MACVO) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract body-frame poses and timestamps from the visual map.

    Returns:
        poses_7     : [N, 7] float32 SE3 [x, y, z, qx, qy, qz, qw]
        timestamps_ns: [N] int64
    """
    global_map = system.get_map()
    sensor_poses = pp.SE3(global_map.frames.data["pose"].tensor)
    T_BS = pp.SE3(global_map.frames.data["T_BS"].tensor)
    body_poses = (T_BS @ sensor_poses @ T_BS.Inv()).tensor().cpu().numpy()
    timestamps_ns = global_map.frames.data["time_ns"].tensor.cpu().numpy().astype(np.int64)
    return body_poses, timestamps_ns


# ── Rerun visualization callback (mirrors MACVO.py) ───────────────────────────

def _visualize(frame: StereoFrame, system: MACVO) -> None:
    rr.set_time_sequence("frame_idx", frame.frame_idx)
    if system.graph.frames.data["need_interp"][-1]:
        return  # non-keyframe interpolated pose — skip heavy logging
    if frame.frame_idx > 0:
        rr_plt.log_trajectory("/world/est", pp.SE3(system.graph.frames.data["pose"].tensor))
    rr_plt.log_camera("/world/macvo/cam_left",
                      pp.SE3(system.graph.frames.data["pose"][-1]),
                      system.graph.frames.data["K"][-1])
    rr_plt.log_image("/world/macvo/cam_left", frame.stereo.imageL[0].permute(1, 2, 0))
    map_points = system.graph.get_frame2map(system.graph.frames[-1:])
    rr_plt.log_points("/world/point_cloud",
                      map_points.data["pos_Tw"],
                      map_points.data["color"],
                      map_points.data["cov_Tw"], "sphere")


# ── Core sequence runner ───────────────────────────────────────────────────────

def run_sequence(bagfiles: list[str], seq: str, round_dir: Path, args) -> None:
    """
    Run MAC-VO on one sequence and save trajectory + latency.
    Uses a manual frame loop (instead of receive_frames) to capture per-frame timing.
    """
    round_dir.mkdir(parents=True, exist_ok=True)
    traj_file = round_dir / f"{seq}_AllFrameTrajectory.txt"
    latency_file = round_dir / f"{seq}_latency.txt"

    # Load odometry config
    odom_cfg, odom_cfg_dict = load_config(Path(args.odom))
    project_name = odom_cfg.Odometry.name + "@" + seq

    if args.useRR:
        # spawn=False: don't try to launch a viewer process inside the container.
        # Connect to whatever address --rr_addr points at (host viewer over TCP).
        rr.init(project_name, spawn=False)
        rr.connect_tcp(args.rr_addr)
        rr.log("/", rr.ViewCoordinates(xyz=rr.ViewCoordinates.FRD), static=True)
        rr_plt.default_mode = "rerun"

    exp_space = Sandbox.create(Path(round_dir, "sandbox"), project_name)
    exp_space.config = {
        "Project": project_name,
        "Odometry": odom_cfg_dict["Odometry"],
        "Data": {"seq": seq},
    }

    Timer.setup(active=args.timing)
    system: MACVO = MACVO[StereoFrame].from_config(asNamespace(exp_space.config))

    # Scale intrinsics proportionally when image is downscaled
    s = args.img_scale
    K_scaled = TSRB_K.clone()
    K_scaled[0, 0, 0] *= s   # fx
    K_scaled[0, 1, 1] *= s   # fy
    K_scaled[0, 0, 2] *= s   # cx
    K_scaled[0, 1, 2] *= s   # cy
    w_scaled = int(TSRB_WIDTH * s)
    h_scaled = int(TSRB_HEIGHT * s)

    stream = RosbagStereoStream(bagfiles, stride=args.stride, img_scale=s)
    n_approx = len(stream)
    print(f"  ~{n_approx} stereo frames from {[Path(b).name for b in bagfiles]}")
    if s != 1.0:
        print(f"  Image scale: {s}  ({w_scaled}×{h_scaled})")
    print(f"  Warmup frames (excluded from stats): {args.warmup}")

    latency_records: list[tuple[float, float]] = []

    try:
        for frame_idx, time_ns, imageL, imageR in tqdm(stream, total=n_approx, desc=seq):
            frame = StereoFrame(
                idx=[frame_idx],
                time_ns=[time_ns],
                stereo=StereoData(
                    T_BS=TSRB_T_BS,
                    K=K_scaled,
                    baseline=torch.tensor([TSRB_BASELINE]),
                    width=w_scaled,
                    height=h_scaled,
                    time_ns=[time_ns],
                    imageL=imageL,
                    imageR=imageR,
                ),
                gt_pose=None,
            )

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            system.run(frame)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            # Skip warmup frames from the saved latency (CUDA graph compilation, JIT)
            if frame_idx >= args.warmup:
                latency_records.append((time_ns * 1e-9, elapsed))

            if args.useRR:
                _visualize(frame, system)

        system.terminate()

    except KeyboardInterrupt:
        system.terminate()
        print("\nInterrupted — saving partial results.")

    if args.timing:
        Timer.report()
        Timer.save_elapsed(exp_space.path("elapsed_time.json"))

    body_poses, timestamps_ns = extract_trajectory(system)
    save_tum_trajectory(body_poses, timestamps_ns, traj_file)
    save_latency(latency_records, latency_file)

    del system
    torch.cuda.empty_cache()

    if latency_records:
        times_ms = np.array([r[1] for r in latency_records]) * 1000
        print(f"  Latency (excl. {args.warmup} warmup frames):")
        print(f"    mean   {np.mean(times_ms):.1f} ms  ({1000/np.mean(times_ms):.1f} fps)")
        print(f"    median {np.median(times_ms):.1f} ms  ({1000/np.median(times_ms):.1f} fps)")
        print(f"    p95    {np.percentile(times_ms, 95):.1f} ms")
    else:
        print("  No frames processed after warmup.")
    print(f"  Saved → {traj_file}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run MAC-VO on TSRB stereo rosbags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--odom", default="Config/Experiment/MACVO/MACVO_Fast.yaml",
                   help="Odometry config YAML (relative to MAC-VO root)")
    p.add_argument("--bag_root", default=BAG_ROOT,
                   help="Root directory containing per-sequence bag subdirectories")
    p.add_argument("--bagfile", default=None,
                   help="Single bag path — skips dataset discovery")
    p.add_argument("--seq", default=None,
                   help="Sequence name: output label with --bagfile, or filter in dataset mode")
    p.add_argument("--result_dir", default="/mnt/DATA/experiments/mac_vo/")
    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--stride", type=int, default=1,
                   help="Frame stride (1 = every frame, 2 = every other frame)")
    p.add_argument("--img_scale", type=float, default=1.0,
                   help="Downscale factor for images and intrinsics (e.g. 0.5 → 320×240)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Frames to exclude from latency stats (CUDA graph compilation happens here)")
    p.add_argument("--timing", action="store_true",
                   help="Enable MAC-VO internal Timer for per-module breakdown (saved to elapsed_time.json)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-run even if trajectory file already exists")
    p.add_argument("--useRR", action="store_true",
                   help="Stream live visualization to a Rerun viewer over TCP")
    p.add_argument("--rr_addr", default="127.0.0.1:9876",
                   help="Rerun viewer TCP address (use host IP when running inside Docker)")
    return p.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)

    method_name = "mac_vo"

    # ── Single bag mode ────────────────────────────────────────────────────────
    if args.bagfile is not None:
        seq = args.seq or Path(args.bagfile).stem
        round_dir = result_dir / "tsrb" / method_name / "Round1"
        run_sequence([args.bagfile], seq, round_dir, args)
        return

    # ── Multi-sequence mode ────────────────────────────────────────────────────
    sequences = TSRB_SEQUENCES
    if args.seq is not None:
        if args.seq not in sequences:
            print(f"Sequence '{args.seq}' not in dataset. Available:")
            for s in sequences:
                print(f"  {s}")
            return
        sequences = [args.seq]

    for seq in sequences:
        try:
            bagfiles = find_bagfiles(seq, args.bag_root)
        except FileNotFoundError as e:
            print(f"  Skipping {seq}: {e}")
            continue

        for round_idx in range(1, args.num_rounds + 1):
            print(f"\n=== mac_vo | tsrb | {seq} | Round {round_idx} ===")
            round_dir = result_dir / "tsrb" / method_name / f"Round{round_idx}"
            traj_file = round_dir / f"{seq}_AllFrameTrajectory.txt"

            if not args.overwrite and traj_file.is_file():
                print("  Already done, skipping.")
                continue

            try:
                run_sequence(bagfiles, seq, round_dir, args)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                return
            except Exception as e:
                print(f"  [ERROR] {seq} round {round_idx}: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
