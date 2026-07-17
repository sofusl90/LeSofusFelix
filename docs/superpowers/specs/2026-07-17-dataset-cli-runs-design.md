# Dataset switching, run management CLI, and LeWM tworooms support

*Approved approach: run-centric layout ("approach A"), 2026-07-17.*

## Problem

Two entangled gaps:

1. Training data is hardcoded to Breakout videos with all-zero actions, so the
   predictor's AdaLN action conditioning has never received a real signal. The
   LeWM paper's TwoRoom dataset (downloaded to `data/lewm-tworooms/tworoom.h5`)
   provides pixels + real continuous actions.
2. `train()` auto-resumes from a single global `checkpoints/` dir whenever any
   checkpoint exists. Switching datasets would silently restore old weights and
   optimizer state into the new run.

A run is only meaningful as (weights + config + dataset) together, so dataset
switching and resume-vs-fresh are one design: runs become self-contained
directories, and the CLI selects a dataset preset and a run.

## CLI

`main.py` grows an argparse interface; presets stay greppable in `main.py`
(the run dir records which preset and its values, but code is the source of
truth on resume — accepted trade-off for simplicity).

```
python main.py <dataset>              # new run, auto-named <dataset>_<timestamp>
python main.py <dataset> --run NAME   # named run: create if missing, else resume
```

- `<dataset>` ∈ `{breakout, tworooms}`.
- New run: create `runs/NAME/`, write `config.json` (dataset name + full config
  values as a record), train from scratch.
- Resume: `runs/NAME/` exists → error if `config.json`'s dataset ≠ CLI dataset;
  otherwise restore the latest checkpoint in that run and continue at the next
  epoch (existing epoch-granularity resume logic, now scoped per run).

## Run layout

```
runs/<name>/
  config.json        # dataset + config snapshot, written at creation
  checkpoints/       # orbax CheckpointManager dir (max_to_keep=3)
  plots/             # loss.png, recon.png
```

`train(model, config, dataloader, key, run_dir)` takes the run dir; the
`CHECKPOINT_DIR` global and timestamped `plots/` dirs go away. Existing root
`checkpoints/` and `plots/` are left untouched (legacy); new work happens under
`runs/`.

## Datasets

Both datasets are normalized by one-time build functions into the same on-disk
triple under `data/<dataset>/`, consumed by a single `Dataloader`:

```
data/<dataset>/frames.npy    # (N, 224, 224, 3) uint8, memmap'd at load
data/<dataset>/actions.npy   # (N, A) float32 — action block from frame i to i+1
data/<dataset>/starts.npy    # window starts that stay within one episode/clip
```

- **breakout** (existing video path): `build_dataset` moves its outputs to
  `data/breakout/`; `actions.npy` is zeros with A=1 (no recorded actions).
  Existing root `frames.npy`/`starts.npy` can be moved there to skip a rebuild.
- **tworooms** (new): read `tworoom.h5` (`h5py` + `hdf5plugin`, imported only
  inside the build function) sequentially — random access into the Blosc
  chunks is ~5.5 s per batch, hence the one-time re-materialization. Apply the
  paper's frame-skip 5 per episode: keep frames at in-episode offsets 0, 5, 10,
  …; `actions.npy[i]` = the 5 raw 2-D actions between kept frame i and i+1,
  concatenated (A=10). The episode-terminal NaN action is never included by
  construction (it has no successor frame). Trailing steps that don't fill a
  5-action block are dropped. Result: ~184k frames (~27.7 GB), ~154k windows.
  Window starts are stride-aligned — 5× fewer than the paper's arbitrary
  offsets, still exceeding the ~128k samples a 10-epoch run draws.

`Dataloader` keeps its current contract (re-iterable, batches a pure function
of (seed, epoch)) but reads the triple from `data/<dataset>/` and yields real
actions.

## Presets

Per-dataset values in `main.py`; everything not listed is shared:

| | breakout | tworooms |
|---|---|---|
| seq_len | 8 | 4 |
| batch_size | 32 | 128 |
| action_dim | 1 | 10 |
| sigreg_lambda | 0.4 | 0.1 (paper) |

tworooms matches the LeWM paper's TwoRoom recipe (224×224, T=4, B=128,
frame-skip 5 with action blocks, λ=0.1, per-timestep SIGReg) so results are
comparable against the paper and its published checkpoints; if collapse
persists at the paper's own settings on its own data, the bug is in our
implementation (prime suspect: the paper's post-ViT projection uses BatchNorm
and calls it necessary for SIGReg — check `encoder.py`'s projector; that check
is part of this work's verification, not a code change committed blindly).

## Dependencies

`h5py` and `hdf5plugin` added via `uv add` (build-time only imports).

## Error handling

- Unknown dataset name → argparse choices error.
- Resume with mismatched dataset → clear error naming the run's dataset.
- Missing source data (no `videos/`, no `tworoom.h5`) → existing/explicit
  `ValueError` from the build functions.

## Testing

- Build functions: episode-boundary and NaN-exclusion invariants checked on the
  real h5 (spot-check a few episodes: frame indices, action-block alignment,
  no NaNs in `actions.npy`, no window crossing an episode).
- CLI: new run creates the layout; rerun with same `--run` resumes at the next
  epoch; mismatched dataset errors.
- End-to-end: short tworooms training run; watch `copy` vs `pred`, `rank`,
  `tvar` for the collapse signature.
