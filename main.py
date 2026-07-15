import jax
from flax import nnx

from data import Dataloader
from encoder import EncoderConfig
from predictor import PredictorConfig
from train import LeWM, TrainConfig, train

STATE_DIM = 64
ENCODER_DIM = 64
LATENT_DIM = STATE_DIM + ENCODER_DIM


predicter_config = PredictorConfig(
    latent_dim=LATENT_DIM,
    state_dim=STATE_DIM,
    action_dim=1,
    num_state_tokens=8,
    mlp_ratio=4,
    num_blocks=4,
    dropout_rate=0.1,
)

encoder_config = EncoderConfig(
    image_size=224,
    patch_size=14,
    in_channels=3,
    hidden_size=192,
    num_heads=3,
    encoder_dim=ENCODER_DIM,
    state_dim=STATE_DIM,
    mlp_ratio=1,
    num_blocks=4,
    dropout_rate=0.1,
)

train_config = TrainConfig(
    adamw_lr=3e-4,
    epochs=100,
    batch_size=8,
    seq_len=16,
)


def run():
    model = LeWM(encoder_config, predicter_config, rngs=nnx.Rngs(0))
    key = jax.random.key(42)
    train(model, train_config, Dataloader(train_config.batch_size, train_config.seq_len), key)


if __name__ == "__main__":
    run()
