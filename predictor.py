import jax.numpy as jnp
from flax import nnx
import jax

from dataclasses import dataclass


@dataclass
class PredictorConfig:
    latent_dim: int
    state_dim: int
    action_dim: int
    num_state_tokens: int
    mlp_ratio: int
    num_blocks: int
    dropout_rate: float


class PredictorBlock(nnx.Module):
    def __init__(self, config: PredictorConfig, rngs: nnx.Rngs):
        qkv_dim = max(config.latent_dim, config.state_dim)
        mlp_ratio = config.mlp_ratio

        self.z_norm = nnx.LayerNorm(config.latent_dim, rngs=rngs)
        self.state_norm = nnx.LayerNorm(config.state_dim, rngs=rngs)

        self.state_cross_attn = nnx.MultiHeadAttention(
            num_heads=8, in_features=config.state_dim, in_kv_features=config.latent_dim,
            qkv_features=qkv_dim, out_features=config.state_dim, decode=False, rngs=rngs)

        self.z_pred_attn = nnx.MultiHeadAttention(
            num_heads=8, in_features=config.latent_dim, in_kv_features=config.state_dim,
            qkv_features=qkv_dim, out_features=config.latent_dim, decode=False, rngs=rngs)

        self.state_mlp_norm = nnx.LayerNorm(config.state_dim, rngs=rngs)
        self.state_mlp_fc1 = nnx.Linear(config.state_dim, config.state_dim * mlp_ratio, rngs=rngs)
        self.state_mlp_fc2 = nnx.Linear(config.state_dim * mlp_ratio, config.state_dim, rngs=rngs)

        self.z_mlp_norm = nnx.LayerNorm(config.latent_dim, rngs=rngs)
        self.z_mlp_fc1 = nnx.Linear(config.latent_dim, config.latent_dim * mlp_ratio, rngs=rngs)
        self.z_mlp_fc2 = nnx.Linear(config.latent_dim * mlp_ratio, config.latent_dim, rngs=rngs)

    def __call__(self, z, state):
        # z: (B, Nz, latent_dim), state: (B, Ns, state_dim)
        z_n = self.z_norm(z)
        state_n = self.state_norm(state)
        state = state + self.state_cross_attn(state_n, z_n)

        state_n2 = self.state_mlp_norm(state)
        state = state + self.state_mlp_fc2(nnx.gelu(self.state_mlp_fc1(state_n2)))

        z_n2 = self.z_norm(z)
        state_n3 = self.state_norm(state)
        z = z + self.z_pred_attn(z_n2, state_n3)

        z_n3 = self.z_mlp_norm(z)
        z = z + self.z_mlp_fc2(nnx.gelu(self.z_mlp_fc1(z_n3)))

        return z, state


class Predictor(nnx.Module):
    def __init__(self, config: PredictorConfig, rngs: nnx.Rngs):
        self.config = config

        self.action_embed = nnx.Linear(config.action_dim, config.latent_dim, rngs=rngs)
        self.blocks = nnx.List([
            PredictorBlock(config, rngs=rngs)
            for _ in range(config.num_blocks)
        ])

    def __call__(self, z: jax.Array, state: jax.Array, action: jax.Array):
        # z: (B, Nz, latent_dim), state: (B, Ns, state_dim), action: (B, action_dim)
        a = self.action_embed(action)[:, None, :]     # (B, 1, latent_dim)
        z = z + a

        for block in self.blocks:
            z, state = block(z, state)

        return z, state
