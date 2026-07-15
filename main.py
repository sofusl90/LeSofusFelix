import jax.numpy as jnp
import optax

STATE_DIM = 64
ENCODER_DIM = 64
LATENT_DIM = STATE_DIM + ENCODER_DIM


predicter_config = {
    "latent_dim": LATENT_DIM,
    "state_dim": STATE_DIM,
    "num_blocks": 4,
    "dropout_rate": 0.1,
}

encoder_config = {
    "image_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "hidden_size": 192,
    "num_heads": 3,
    "latent_dim": ENCODER_DIM,
    "state_dim": STATE_DIM,
    "mlp_ratio": 1,
    "num_blocks": 4,
    "dropout_rate": 0.1,
}
