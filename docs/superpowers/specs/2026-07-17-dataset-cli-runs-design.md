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

`main.py` grows an argparse interface; per-dataset presets stay greppable in
`main.py` as defaults, overridable per run via a config file.

```
python main.py <dataset>                     # new run, auto-named <dataset>_<timestamp>
python main.py <dataset> --run NAME          # named run: create if missing, else resume
python main.py <dataset> --config over.json  # new run with hyperparameter overrides
```

- `<dataset>` ∈ `{breakout, tworooms}`.
- `--config` is a flat JSON dict of `TrainConfig` field overrides (e.g.
  `{"adamw_lr": 1e-4, "sigreg_lambda": 0.2}`), merged over the dataset preset
  at run creation. Unknown keys are an error. Architecture configs stay in
  code — a config file can't change model shape, with one deliberate
  exception: the model configs are derived from the resolved `TrainConfig`
  where they overlap (`seq_len` → predictor positional embeddings,
  `action_dim` from the dataset), so overrides propagate consistently.
- New run: create `runs/NAME/`, resolve preset + overrides, write the resolved
  values to `config.json` (dataset name + full `TrainConfig`), train from
  scratch.
- Resume: `runs/NAME/` exists → error if `config.json`'s dataset ≠ CLI dataset;
  `--config` alongside an existing run is also an error (a run's
  hyperparameters are fixed at creation). The run's `config.json` is the
  source of truth on resume — code presets are only defaults for new runs, so
  a resumed run never silently drops its overrides. Restore the latest
  checkpoint and continue at the next epoch (existing epoch-granularity resume
  logic, now scoped per run).

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

## Dataset format (the contract)

Every dataset — current and future — is normalized by a one-time build
function into the same on-disk layout under `data/<dataset>/`, and a single
`Dataloader` consumes only this layout (it never touches videos or h5 files):

```
data/<dataset>/frames.npy    # (N, 224, 224, 3) uint8, memmap'd at load
data/<dataset>/actions.npy   # (N, A) float32 — action taken between frame i and i+1
data/<dataset>/episodes.npy  # (E+1,) int64 — cumulative episode frame offsets
data/<dataset>/meta.json     # {"action_dim": A, "num_frames": N, "source": ...}
```

Actions are mandatory: a dataset without recorded actions must still emit
`actions.npy`, filled with zeros. `action_dim` lives in `meta.json` and flows
from there into the predictor config — it is a property of the dataset, not a
preset value, so the model and data can never disagree on it.

Window starts are derived from `episodes.npy` by the `Dataloader` at load time,
not baked at build time: baked starts would encode one `seq_len`, which
`--config` can override, silently training on stale windows. The layout is thus
seq_len-independent. The loader yields actions as `(B, T-1, A)` — the
transitions inside the window — so an episode's final frame (whose outgoing
block is the zero filler) is never consumed.

- **breakout** (existing video path): `build_breakout` writes its outputs under
  `data/breakout/` and emits zero actions with A=1, with one episode span per
  source clip.
- **tworooms** (new): read `tworoom.h5` (`h5py` + `hdf5plugin`, imported only
  inside the build function) sequentially — random access into the Blosc
  chunks is ~5.5 s per batch, hence the one-time re-materialization. Apply the
  paper's frame-skip 5 per episode: keep frames at in-episode offsets 0, 5, 10,
  …; `actions.npy[i]` = the 5 raw 2-D actions between kept frame i and i+1,
  concatenated (A=10). The episode-terminal NaN action is never included by
  construction (it has no successor frame). Trailing steps that don't fill a
  5-action block leave a zero-filled row that the loader never reads. Actual
  build: 190,562 frames (~27 GB), 10,000 episodes, 160,562 windows at T=4.

`Dataloader` keeps its current contract (re-iterable, batches a pure function
of (seed, epoch)) but reads from `data/<dataset>/` and yields real actions.

## Presets

Per-dataset values in `main.py`; everything not listed is shared:

| | breakout | tworooms |
|---|---|---|
| seq_len | 8 | 4 |
| batch_size | 32 | 128 |
| sigreg_lambda | 0.4 | 0.1 (paper) |

(`action_dim` is not a preset — it comes from the dataset's `meta.json`:
1 for breakout, 10 for tworooms.)

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
- Unknown keys in a `--config` file → error listing the offending keys.
- `--config` passed together with an existing run → error (hyperparameters are
  fixed at run creation).
- Resume with mismatched dataset → clear error naming the run's dataset.
- Missing source data (no `videos/`, no `tworoom.h5`) → existing/explicit
  `ValueError` from the build functions.

## Testing

- Build functions: episode-boundary and NaN-exclusion invariants checked on the
  real h5 (spot-check a few episodes: frame indices, action-block alignment,
  no NaNs in `actions.npy`, no window crossing an episode).
- CLI: new run creates the layout; rerun with same `--run` resumes at the next
  epoch with the run's own `config.json` values (not code defaults);
  mismatched dataset errors; `--config` overrides land in the snapshot and
  unknown keys error.
- End-to-end: short tworooms training run; watch `copy` vs `pred`, `rank`,
  `tvar` for the collapse signature.
