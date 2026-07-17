import jax
import jax.numpy as jnp
from flax import nnx

from data import Dataloader
from decoder import DecoderConfig
from encoder import EncoderConfig
from predictor import PredictorConfig
from train import LeWM, TrainConfig, train


LATENT_DIM = 96
SEQ_LEN = 4

predicter_config = PredictorConfig(
    latent_dim=LATENT_DIM,
    action_dim=1,
    num_heads=8,
    dim_head=64,
    mlp_dim=1024,
    num_blocks=6,
    dropout_rate=0.1,
    seq_len=SEQ_LEN,
    proj_hidden_dim=1024,
    dtype=jnp.bfloat16,
)

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

decoder_config = DecoderConfig(
    latent_dim=LATENT_DIM,
    image_size=224,
    base_size=7,
    base_channels=256,
    stage_channels=(128, 64, 32, 16, 8),
    dtype=jnp.bfloat16,
)

train_config = TrainConfig(
    adamw_lr=5e-5,
    epochs=100,
    batch_size=32,
    seq_len=SEQ_LEN,
    sigreg_lambda=0.4,
    recon_lambda=1.0,
    weight_decay=1e-3,
    grad_clip=1.0,
    warmup_steps=500,
)


def run():
    model = LeWM(encoder_config, predicter_config, decoder_config, rngs=nnx.Rngs(0))
    key = jax.random.key(42)
    train(model, train_config, Dataloader(train_config.batch_size, train_config.seq_len), key)


if __name__ == "__main__":
    run()
