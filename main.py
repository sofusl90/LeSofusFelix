from flax.nnx.filterlib import Predicate
import jax.numpy as jnp
import optax
from flax import nnx
from train import TrainConfig
from train import train, LeWM
from data import dataloader


STATE_DIM = 64
ENCODER_DIM = 64
LATENT_DIM = STATE_DIM + ENCODER_DIM


predicter_config = PredictorConfig(
    latent_dim = LATENT_DIM,
    state_dim = STATE_DIM,
    num_blocks = 4,
    dropout_rate = 0.1,
)

encoder_config = EncoderConfig(
    image_size = 224,
    patch_size = 14,
    in_channels = 3,
    hidden_size = 192,
    num_heads = 3,
    encoder_dim = ENCODER_DIM,
    state_dim = STATE_DIM,
    mlp_ratio = 1,
    num_blocks = 4,
    dropout_rate = 0.1,
)

train_config = TrainConfig(
    adamw_lr=3e-4,
    epochs=100,
)


def run():

    model = LeWM(enc_config, pred_config, rngs=nnx.Rngs(0))
    train(model, train_config, dataloader(batch_size=8, seq_len=16), key)


if __name__ == "__main__":
    run()
