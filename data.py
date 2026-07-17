import json
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np

DATA_ROOT = Path("data")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def probe_fps(video):
    """Native frame rate of `video`, as a float."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1", str(video),
    ]
    num, den = subprocess.run(cmd, capture_output=True, check=True, text=True).stdout.split("/")
    return int(num) / int(den)


def decode_video(video, skip=50.0, fps=None, size=224):
    """Decode a video from `skip` seconds onward into a (N, size, size, 3) uint8 array.
    Defaults to the video's native frame rate, capped at 60.
    """
    fps = fps or min(probe_fps(video), 60)
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(skip), "-i", str(video),
        "-vf", f"fps={fps},scale={size}:{size}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def write_meta(root, action_dim, num_frames, source):
    meta = {"action_dim": action_dim, "num_frames": num_frames, "source": source}
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def build_breakout(root, video_dir="videos"):
    """Decode every video under `video_dir` into the dataset contract. No actions
    were recorded, so every transition gets a single zero action.
    """
    videos = sorted(p for p in Path(video_dir).iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise ValueError(f"no videos found in {video_dir}")

    clips = []
    for video in videos:
        clips.append(decode_video(video))
        print(f"{video.name}: {len(clips[-1])} frames")

    frames = np.concatenate(clips)
    np.save(root / "frames.npy", frames)
    np.save(root / "actions.npy", np.zeros((len(frames), 1), np.float32))
    np.save(root / "episodes.npy", np.cumsum([0] + [len(c) for c in clips], dtype=np.int64))
    write_meta(root, 1, len(frames), video_dir)


def build_tworooms(root, h5_path="data/lewm-tworooms/tworoom.h5", frame_skip=5):
    """Re-materialize the LeWM TwoRoom h5 into the dataset contract: keep every
    `frame_skip`-th frame per episode and concatenate the raw actions in between
    into one block per kept transition (the paper's action-block scheme). The
    h5's Blosc-compressed chunks are too slow for random access at train time,
    hence this one-time sequential pass. Each episode's terminal NaN action is
    never read: the last block ends one step before the final kept frame.
    """
    import h5py
    import hdf5plugin  # noqa: F401 -- registers the Blosc filter h5py lacks

    f = h5py.File(h5_path, "r", rdcc_nbytes=32 * 2**20)
    pixels = f["pixels"]
    action = f["action"][:].astype(np.float32)
    offsets, lengths = f["ep_offset"][:], f["ep_len"][:]

    kept = [np.arange(off, off + ln, frame_skip) for off, ln in zip(offsets, lengths)]
    episodes = np.cumsum([0] + [len(k) for k in kept], dtype=np.int64)
    total = int(episodes[-1])
    block_dim = action.shape[1] * frame_skip

    frames = np.lib.format.open_memmap(
        root / "frames.npy", mode="w+", dtype=np.uint8, shape=(total, *pixels.shape[1:])
    )
    actions = np.zeros((total, block_dim), np.float32)
    for i, (idx, off, ln) in enumerate(zip(kept, offsets, lengths)):
        frames[episodes[i] : episodes[i + 1]] = pixels[off : off + ln][::frame_skip]
        blocks = action[idx[0] : idx[-1]].reshape(-1, block_dim)
        actions[episodes[i] : episodes[i] + len(blocks)] = blocks
        if i % 500 == 0:
            print(f"episode {i}/{len(kept)}")
    frames.flush()
    assert np.isfinite(actions).all()

    np.save(root / "actions.npy", actions)
    np.save(root / "episodes.npy", episodes)
    write_meta(root, block_dim, total, str(h5_path))


BUILDERS = {"breakout": build_breakout, "tworooms": build_tworooms}


class Dataloader:
    """Yields `num_batches` random (obs, actions) windows per iteration from
    `data/<dataset>/`, building the dataset from its source on first use.
    obs is (B, T, H, W, C) in [0, 1]; actions is (B, T-1, A) where actions[:, i]
    leads from frame i to i+1, so a window never needs its final frame's block.
    Window starts are derived from episode boundaries here rather than baked at
    build time, keeping the on-disk layout independent of seq_len. Batches are
    a pure function of (seed, epoch), so set `epoch` before iterating and a
    resumed run samples exactly what an uninterrupted one would have.
    """

    def __init__(self, dataset, batch_size, seq_len, num_batches=100, seed=0):
        root = DATA_ROOT / dataset
        if not (root / "meta.json").exists():
            root.mkdir(parents=True, exist_ok=True)
            BUILDERS[dataset](root)
        self.action_dim = json.loads((root / "meta.json").read_text())["action_dim"]
        self.frames = np.load(root / "frames.npy", mmap_mode="r")
        self.actions = np.load(root / "actions.npy")
        episodes = np.load(root / "episodes.npy")
        self.starts = np.concatenate([
            np.arange(a, b - seq_len + 1)
            for a, b in zip(episodes[:-1], episodes[1:]) if b - a >= seq_len
        ])
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.num_batches):
            starts = rng.choice(self.starts, size=self.batch_size)
            obs = np.stack([self.frames[s : s + self.seq_len] for s in starts])
            acts = np.stack([self.actions[s : s + self.seq_len - 1] for s in starts])
            yield jnp.asarray(obs, dtype=jnp.float32) / 255.0, jnp.asarray(acts)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "breakout"
    root = DATA_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    BUILDERS[name](root)
    print(f"built {name} under {root}")
