import jax.numpy as jnp
from flax import nnx
import jax

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    image_size: int
    patch_size: int
    in_channels: int
    hidden_size: int
    num_heads: int
    encoder_dim: int
    proj_hidden_dim: int
    mlp_ratio: float
    num_blocks: int
    dropout_rate: float
    dtype: jnp.dtype = jnp.float32

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)



class Projector(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rngs: nnx.Rngs,
                 dtype: jnp.dtype = jnp.float32):
        self.net = nnx.Sequential(
            nnx.Linear(in_dim, hidden_dim, dtype=dtype, rngs=rngs),
            nnx.BatchNorm(hidden_dim, dtype=dtype, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, out_dim, dtype=dtype, rngs=rngs),
        )

    def __call__(self, x):
        return self.net(x)


class EncoderBlock(nnx.Module):
    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, dtype=config.dtype, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.hidden_size,
            qkv_features=config.hidden_size,
            decode=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(config.hidden_size, dtype=config.dtype, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(config.hidden_size, config.mlp_dim, dtype=config.dtype, rngs=rngs),
            nnx.gelu,
            nnx.Linear(config.mlp_dim, config.hidden_size, dtype=config.dtype, rngs=rngs),
        )

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nnx.Module):
    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.patch_embeddings = nnx.Conv(
                    config.in_channels,
                    config.hidden_size,
                    kernel_size=(config.patch_size, config.patch_size),
                    strides=(config.patch_size, config.patch_size),
                    padding="VALID",
                    use_bias=True,
                    dtype=config.dtype,
                    rngs=rngs,
                )

        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nnx.Param(
            jnp.zeros((1, 1, config.hidden_size))
        )

        self.pos_embedding = nnx.Param(
            nnx.initializers.truncated_normal(stddev=0.02)(
                rngs.params(), (1, num_patches + 1, config.hidden_size)
            )
        )

        self.blocks = nnx.List([
            EncoderBlock(config, rngs=rngs)
            for _ in range(config.num_blocks)
        ])

        self.final_norm = nnx.LayerNorm(config.hidden_size, dtype=config.dtype, rngs=rngs)

        self.projector = Projector(
            config.hidden_size, config.proj_hidden_dim, config.encoder_dim,
            rngs=rngs, dtype=config.dtype,
        )


    def __call__(self, x: jax.Array):                      # (B, H, W, C)
        x = self.patch_embeddings(x)                       # (B, H/P, W/P, D)
        B, h, w, D = x.shape
        x = x.reshape(B, h * w, D)                         # (B, N, D)

        cls = jnp.broadcast_to(self.cls_token.value, (B, 1, D))
        x = jnp.concatenate([cls, x], axis=1)              # (B, N+1, D)
        x = x + self.pos_embedding.value

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x[:, 0]                                        # (B, D)

        x = self.projector(x)                              # (B, encoder_dim)
        return x




def sigreg(proj, key, num_proj=1024, knots=17, t_max=3.0):
    # proj: (..., B, d)
    *_, B, d = proj.shape

    t = jnp.linspace(0.0, t_max, knots, dtype=jnp.float32)                          # (knots,)
    dt = t_max / (knots - 1)
    w = jnp.full((knots,), 2 * dt, dtype=jnp.float32).at[0].set(dt).at[-1].set(dt)  # (knots,)
    phi = jnp.exp(-0.5 * jnp.square(t))                                             # (knots,)
    weights = w * phi

    A = jax.random.normal(key, (d, num_proj))          # (d, num_proj)
    A = A / jnp.linalg.norm(A, axis=0)

    x_t = (proj @ A)[..., None] * t                    # (..., B, num_proj, knots)
    err = jnp.square(jnp.cos(x_t).mean(-3) - phi) + jnp.square(jnp.sin(x_t).mean(-3))  # (..., num_proj, knots)
    statistic = (err @ weights) * B                    # (..., num_proj)

    return statistic.mean()                            # scalar
