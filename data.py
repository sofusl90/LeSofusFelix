import subprocess

import jax.numpy as jnp
import numpy as np

FRAMES_PATH = "frames.npy"


def extract_frames(video, skip=50.0, fps=10, size=224):
    """Decode video from `skip` seconds onward into a (N, size, size, 3) uint8 array."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(skip), "-i", video,
        "-vf", f"fps={fps},scale={size}:{size}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)
    np.save(FRAMES_PATH, frames)
    return frames


def dataloader(batch_size, seq_len, action_dim=1, num_batches=1000, seed=0):
    frames = np.load(FRAMES_PATH, mmap_mode="r")
    rng = np.random.default_rng(seed)
    actions = jnp.zeros((batch_size, seq_len, action_dim))

    for _ in range(num_batches):
        starts = rng.integers(0, len(frames) - seq_len, batch_size)
        obs = np.stack([frames[s : s + seq_len] for s in starts])
        yield jnp.asarray(obs, dtype=jnp.float32) / 255.0, actions


if __name__ == "__main__":
    frames = extract_frames("breakout.mp4")
    print(f"saved {frames.shape} to {FRAMES_PATH}")
