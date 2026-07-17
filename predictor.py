from flax import nnx
import jax
import jax.numpy as jnp

from dataclasses import dataclass


def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array):
    return x * (1 + scale) + shift



@dataclass
class PredictorConfig:   # The paper uses these configs
    latent_dim: int      # 192
    action_dim: int
    num_heads: int       # 16
    dim_head: int        # 64
    mlp_dim: int         # 2048
    num_blocks: int      # 6
    dropout_rate: float  # 0.1
    seq_len: int
    proj_hidden_dim: int # 2048
    dtype: jnp.dtype = jnp.float32


class PredictorBlock(nnx.Module):
    def __init__(self, config: PredictorConfig, rngs: nnx.Rngs):

        self.norm1 = nnx.LayerNorm(config.latent_dim, use_scale=False, use_bias=False,
                                   dtype=config.dtype, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.latent_dim,
            qkv_features=config.num_heads * config.dim_head,
            out_features=config.latent_dim,
            decode=False,
            dropout_rate=config.dropout_rate,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(config.latent_dim, use_scale=False, use_bias=False,
                                   dtype=config.dtype, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(config.latent_dim, config.mlp_dim, dtype=config.dtype, rngs=rngs),
            nnx.gelu,
            nnx.Linear(config.mlp_dim, config.latent_dim, dtype=config.dtype, rngs=rngs),
        )

        self.adaLN_modulation = nnx.Linear(
            config.latent_dim,
            6 * config.latent_dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            dtype=config.dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, cs: jax.Array):
        mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(self.adaLN_modulation(jax.nn.silu(cs)), 6, axis=-1)

        residual = x
        x = self.norm1(x)
        x = modulate(x, shift_msa, scale_msa)
        x = residual + gate_msa * self.attn(x, mask=mask)
        residual = x


        x = self.norm2(x)
        x = modulate(x, shift_mlp, scale_mlp)
        x = residual + gate_mlp * self.mlp(x)

        return x


class Predictor(nnx.Module):
    def __init__(self, config: PredictorConfig, rngs: nnx.Rngs):
        self.config = config

        self.action_embed = nnx.Sequential(
            nnx.Linear(config.action_dim, config.latent_dim, dtype=config.dtype, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(config.latent_dim, config.latent_dim, dtype=config.dtype, rngs=rngs),
        )
        self.blocks = nnx.List([
            PredictorBlock(config, rngs=rngs)
            for _ in range(config.num_blocks)
        ])
        self.pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.seq_len, config.latent_dim)) * 0.02
        )
        self.final_norm = nnx.LayerNorm(config.latent_dim, dtype=config.dtype, rngs=rngs)

    def __call__(self, z: jax.Array, cs: jax.Array):
        # z: (B, T, latent_dim), cs: (B, T, action_dim)
        z = z + self.pos_embedding[:, :z.shape[1]]
        cs = self.action_embed(cs)
        for block in self.blocks:
            z = block(z, cs)

        return self.final_norm(z)
