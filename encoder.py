import jax.numpy as jnp
from flax import nnx
import jax


class EncoderConfig:
    image_size: int
    patch_size: int
    in_channels: int
    hidden_size: int
    num_heads: int
    latent_dim: int
    state_dim: int
    mlp_ratio: float
    num_blocks: int
    dropout_rate: float

    @property
    def mlp_dim(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)



class EncoderBlock(nnx.Module):
    def __init__(self, config: EncoderConfig, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.hidden_size,
            qkv_features=config.hidden_size,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(config.hidden_size, rngs=rngs)
        self.mlp = nnx.Sequential(
            nnx.Linear(config.hidden_size, config.mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(config.mlp_dim, config.hidden_size, rngs=rngs),
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

        self.final_norm = nnx.LayerNorm(config.hidden_size, rngs=rngs)

        self.proj_linear = nnx.Linear(config.hidden_size, config.embed_dim, rngs=rngs)
        self.proj_bn = nnx.BatchNorm(config.embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.patch_embeddings(x)
        B, h, w, D = x.shape
        x = x.reshape(B, h * w, D)

        cls = jnp.broadcast_to(self.cls_token.value, (B, 1, D))
        x = jnp.concatenate([cls, x], axis=1)
        x = x + self.pos_embedding.value

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x[:, 0]

        x = self.proj_bn(self.proj_linear(x))

        return x
