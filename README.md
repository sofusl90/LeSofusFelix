# LeSofusFelix

A LeJEPA-style world model (ViT encoder + causal predictor, SIGReg anti-collapse)
trained on video with actions.

## Training

```bash
python main.py <dataset> [--run NAME] [--config overrides.json]
```

- `<dataset>` is `breakout` or `tworooms`; it's built from source into
  `data/<dataset>/` on first use.
- `--run NAME` names the run under `runs/NAME/`. Rerunning the same name
  resumes from its latest checkpoint; omit it and a timestamped run is created.
- `--config` points at a JSON file of hyperparameter overrides applied when a
  run is created, e.g. `{"sigreg_lambda": 0.2, "adamw_lr": 1e-4}`. A run's
  hyperparameters are fixed at creation, so `--config` only applies to new runs.

```bash
python main.py tworooms                    # new run, real actions
python main.py tworooms --run baseline     # resume "baseline" if it exists
python main.py tworooms --config over.json # new run with overrides
```

Each run writes checkpoints, plots, and a `config.json` snapshot under
`runs/<name>/`.
