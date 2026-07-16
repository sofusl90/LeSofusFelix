import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np

FRAMES_PATH = "frames.npy"
STARTS_PATH = "starts.npy"
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def decode_video(video, skip=50.0, fps=10, size=224):
    """Decode a video from `skip` seconds onward into a (N, size, size, 3) uint8 array."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(skip), "-i", str(video),
        "-vf", f"fps={fps},scale={size}:{size}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def build_dataset(video_dir, seq_len, skip=50.0, fps=10, size=224):
    """Decode every video under `video_dir` into one concatenated frame array, and
    record which window-start indices stay within a single source clip -- so a
    sampled (seq_len)-frame window never straddles the cut between two videos.
    """
    videos = sorted(p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise ValueError(f"no videos found in {video_dir}")

    clips, starts, offset = [], [], 0
    for video in videos:
        frames = decode_video(video, skip, fps, size)
        if len(frames) > seq_len:
            starts.extend(range(offset, offset + len(frames) - seq_len))
        clips.append(frames)
        offset += len(frames)
        print(f"{video.name}: {len(frames)} frames")

    frames = np.concatenate(clips, axis=0)
    np.save(FRAMES_PATH, frames)
    np.save(STARTS_PATH, np.array(starts, dtype=np.int64))
    return frames


class Dataloader:
    """Yields `num_batches` random windows per iteration; re-iterable across epochs.
    Windows are drawn from `starts.npy`, so they never cross a clip boundary.
    """

    def __init__(self, batch_size, seq_len, action_dim=1, num_batches=100, seed=0):
        if not Path(FRAMES_PATH).exists():
            build_dataset("videos", seq_len)
        self.frames = np.load(FRAMES_PATH, mmap_mode="r")
        self.starts = np.load(STARTS_PATH)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.actions = jnp.zeros((batch_size, seq_len, action_dim))
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.num_batches):
            starts = self.rng.choice(self.starts, size=self.batch_size)
            obs = np.stack([self.frames[s : s + self.seq_len] for s in starts])
            yield jnp.asarray(obs, dtype=jnp.float32) / 255.0, self.actions


if __name__ == "__main__":
    import sys

    video_dir = sys.argv[1] if len(sys.argv) > 1 else "videos"
    frames = build_dataset(video_dir, seq_len=16)
    print(f"saved {frames.shape} frames to {FRAMES_PATH}, {STARTS_PATH} written")
