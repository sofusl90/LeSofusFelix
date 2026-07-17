import argparse
import json
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from data import BUILDERS, Dataloader
from decoder import DecoderConfig
from encoder import EncoderConfig
from predictor import PredictorConfig
from train import LeWM, TrainConfig, train

LATENT_DIM = 96

PRESETS = {
    "breakout": TrainConfig(
        adamw_lr=5e-5, epochs=100, batch_size=32, seq_len=8,
        sigreg_lambda=0.4, recon_lambda=1.0, weight_decay=1e-3,
        grad_clip=1.0, warmup_steps=500,
    ),
    # the LeWM paper's TwoRoom recipe: T=4, B=128, lambda=0.1
    "tworooms": TrainConfig(
        adamw_lr=5e-5, epochs=100, batch_size=128, seq_len=4,
        sigreg_lambda=0.1, recon_lambda=1.0, weight_decay=1e-3,
        grad_clip=1.0, warmup_steps=500,
    ),
}


def make_model(config, action_dim):
    """Model configs are derived from the resolved TrainConfig where they
    overlap (seq_len -> predictor positions) and from the dataset (action_dim),
    so config overrides and data can never disagree with the architecture.
    """
    encoder_config = EncoderConfig(
        image_size=224,
        patch_size=14,
        in_channels=3,
        hidden_size=192,
        num_heads=3,
        encoder_dim=LATENT_DIM,
        proj_hidden_dim=1024,
        mlp_ratio=4,
        num_blocks=8,
        dropout_rate=0.1,
        dtype=jnp.bfloat16,
    )
    predictor_config = PredictorConfig(
        latent_dim=LATENT_DIM,
        action_dim=action_dim,
        num_heads=8,
        dim_head=64,
        mlp_dim=1024,
        num_blocks=6,
        dropout_rate=0.1,
        seq_len=config.seq_len,
        proj_hidden_dim=1024,
        dtype=jnp.bfloat16,
    )
    decoder_config = DecoderConfig(
        latent_dim=LATENT_DIM,
        image_size=224,
        base_size=7,
        base_channels=256,
        stage_channels=(128, 64, 32, 16, 8),
        dtype=jnp.bfloat16,
    )
    return LeWM(encoder_config, predictor_config, decoder_config, rngs=nnx.Rngs(0))


def resolve_run(dataset, run_name, config_path):
    """New runs merge the dataset preset with optional file overrides and
    snapshot the result to runs/<name>/config.json; existing runs are resumed
    from that snapshot alone, so code presets and --config can never silently
    alter a run in flight.
    """
    run_dir = Path("runs") / (run_name or f"{dataset}_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    config_file = run_dir / "config.json"

    if config_file.exists():
        if config_path:
            raise SystemExit(f"{run_dir} already exists; its hyperparameters are fixed")
        saved = json.loads(config_file.read_text())
        if saved["dataset"] != dataset:
            raise SystemExit(f"{run_dir} was created for dataset {saved['dataset']!r}")
        return run_dir, TrainConfig(**saved["train"])

    overrides = json.loads(Path(config_path).read_text()) if config_path else {}
    unknown = set(overrides) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    config = replace(PRESETS[dataset], **overrides)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps({"dataset": dataset, "train": asdict(config)}, indent=2) + "\n")
    return run_dir, config


def run():
    parser = argparse.ArgumentParser(description="Train LeWM on a dataset.")
    parser.add_argument("dataset", choices=sorted(BUILDERS))
    parser.add_argument("--run", help="run name under runs/; resumes if it already exists")
    parser.add_argument("--config", help="JSON file of TrainConfig overrides, new runs only")
    args = parser.parse_args()

    run_dir, config = resolve_run(args.dataset, args.run, args.config)
    print(f"run {run_dir}: {json.dumps(asdict(config))}")
    dataloader = Dataloader(args.dataset, config.batch_size, config.seq_len)
    model = make_model(config, dataloader.action_dim)
    train(model, config, dataloader, jax.random.key(42), run_dir)


if __name__ == "__main__":
    run()
