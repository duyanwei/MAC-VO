"""
TSRB stereo rosbag → MAC-VO frame stream.

Reads synchronized left/right infrared frames from a TSRB ROS bag directly
(no ROS installation needed — uses the pure-Python `rosbags` library).

TSRB stereo topics:
  /camera/infra1/image_rect_raw/compressed  — left  (mono8 JPEG compressed)
  /camera/infra2/image_rect_raw/compressed  — right (mono8 JPEG compressed)

Images are grayscale; they are replicated to 3-channel RGB to match the
Bx3xHxW float32 layout expected by MAC-VO's StereoData.

Camera intrinsics are fixed constants (RealSense D435i @ 640×480) — see
bench_rosbag.py where they are wired into each StereoFrame.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

import cv2
import numpy as np
import torch

from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

_store = get_typestore(Stores.ROS1_NOETIC)

# ── TSRB stereo topic names ────────────────────────────────────────────────────

INFRA1_TOPIC = "/camera/infra1/image_rect_raw/compressed"
INFRA2_TOPIC = "/camera/infra2/image_rect_raw/compressed"

# Maximum timestamp gap (ns) for left/right synchronisation
MAX_SYNC_GAP_NS: int = 50_000_000  # 50 ms


# ── Decode helpers ─────────────────────────────────────────────────────────────

def _decode_mono_jpeg(data_bytes: bytes) -> np.ndarray | None:
    """Decode a mono8 JPEG CompressedImage payload to a uint8 HxW array."""
    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _gray_to_tensor(gray: np.ndarray, scale: float = 1.0) -> torch.Tensor:
    """HxW uint8 gray → 1×3×H×W float32 [0,1] tensor, optionally downscaled."""
    if scale != 1.0:
        h, w = gray.shape
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)   # HxWx3 uint8
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # 1×3×H×W
    t /= 255.0
    return t


# ── Stream ─────────────────────────────────────────────────────────────────────

class RosbagStereoStream:
    """
    Lazy synchronized stereo reader over one or more TSRB rosbag files.

    Bags are read in a single chronological pass.  A small deque buffers
    recent right-camera frames so we can pair each left frame to its
    nearest right neighbour by timestamp.

    Yields: (frame_idx, time_ns, left_tensor, right_tensor)
      - time_ns       : int, nanoseconds
      - left_tensor   : torch.Tensor [1, 3, H, W] float32 RGB in [0,1]
      - right_tensor  : torch.Tensor [1, 3, H, W] float32 RGB in [0,1]
    """

    def __init__(
        self,
        bagfiles: list[str] | str,
        left_topic: str = INFRA1_TOPIC,
        right_topic: str = INFRA2_TOPIC,
        stride: int = 1,
        img_scale: float = 1.0,
    ) -> None:
        self.bagfiles = [bagfiles] if isinstance(bagfiles, str) else list(bagfiles)
        self.left_topic = left_topic
        self.right_topic = right_topic
        self.stride = stride
        self.img_scale = img_scale

        self._approx_len: int | None = None
        self._left_msgtype: str | None = None
        self._right_msgtype: str | None = None

    def _init_msgtypes(self, bag, bagfile: str):
        left_conns = [c for c in bag.connections if c.topic == self.left_topic]
        right_conns = [c for c in bag.connections if c.topic == self.right_topic]
        if not left_conns:
            raise RuntimeError(f"Left topic '{self.left_topic}' not in {bagfile}")
        if not right_conns:
            raise RuntimeError(f"Right topic '{self.right_topic}' not in {bagfile}")
        if self._left_msgtype is None:
            self._left_msgtype = left_conns[0].msgtype
        if self._right_msgtype is None:
            self._right_msgtype = right_conns[0].msgtype
        return left_conns, right_conns

    def _iter_frames(self) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
        frame_idx = 0
        step = 0  # shared across bags so stride is consistent

        for bagfile in self.bagfiles:
            right_buf: deque[tuple[int, bytes]] = deque(maxlen=16)

            with Reader(bagfile) as bag:
                left_conns, right_conns = self._init_msgtypes(bag, bagfile)
                all_conns = left_conns + right_conns

                for conn, ts_ns, rawdata in bag.messages(connections=all_conns):

                    if conn.topic == self.right_topic:
                        msg = _store.deserialize_ros1(rawdata, self._right_msgtype)
                        right_buf.append((ts_ns, msg.data.tobytes()))
                        continue

                    # Left camera frame
                    if step % self.stride != 0:
                        step += 1
                        continue
                    step += 1

                    if not right_buf:
                        continue  # no right frames buffered yet

                    # Find nearest right frame
                    best_dt, best_data = min(
                        ((abs(r_ts - ts_ns), r_data) for r_ts, r_data in right_buf),
                        key=lambda x: x[0],
                    )
                    if best_dt > MAX_SYNC_GAP_NS:
                        continue  # no close match

                    msg_l = _store.deserialize_ros1(rawdata, self._left_msgtype)
                    gray_l = _decode_mono_jpeg(msg_l.data.tobytes())
                    gray_r = _decode_mono_jpeg(best_data)

                    if gray_l is None or gray_r is None:
                        continue

                    yield (frame_idx, int(ts_ns),
                           _gray_to_tensor(gray_l, self.img_scale),
                           _gray_to_tensor(gray_r, self.img_scale))
                    frame_idx += 1

    def __iter__(self):
        return self._iter_frames()

    def __len__(self) -> int:
        """Approximate frame count (left topic messages ÷ stride)."""
        if self._approx_len is None:
            total = 0
            for bagfile in self.bagfiles:
                with Reader(bagfile) as bag:
                    conns = [c for c in bag.connections if c.topic == self.left_topic]
                    if conns:
                        total += sum(c.msgcount for c in conns)
            self._approx_len = max(1, (total + self.stride - 1) // self.stride)
        return self._approx_len
