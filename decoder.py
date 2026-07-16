import jax.numpy as jnp
import jax
from flax import nnx

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    latent_dim: int
    image_size: int
    base_size: int              # spatial resolution right after the latent is projected, e.g. 7
    base_channels: int          # channel count right after the latent is projected, e.g. 256
    stage_channels: tuple       # output channels of each upsampling stage, e.g. (128, 64, 32, 16, 8)
    out_channels: int = 3

    def __post_init__(self):
        num_stages = len(self.stage_channels)
        assert self.base_size * (2 ** num_stages) == self.image_size, (
            f"base_size {self.base_size} * 2^{num_stages} stages must equal image_size {self.image_size}"
        )


class UpBlock(nnx.Module):
    def __init__(self, in_channels: int, out_channels: int, rngs: nnx.Rngs):
        self.conv = nnx.ConvTranspose(
            in_channels, out_channels, kernel_size=(4, 4), strides=(2, 2),
            padding="SAME", rngs=rngs,
        )
        self.norm = nnx.GroupNorm(out_channels, num_groups=min(32, out_channels), rngs=rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return nnx.gelu(x)


class Decoder(nnx.Module):
    def __init__(self, config: DecoderConfig, rngs: nnx.Rngs):
        self.config = config
        self.input_proj = nnx.Linear(
            config.latent_dim, config.base_size * config.base_size * config.base_channels, rngs=rngs
        )

        channels = [config.base_channels, *config.stage_channels]
        self.blocks = nnx.List([
            UpBlock(channels[i], channels[i + 1], rngs=rngs)
            for i in range(len(config.stage_channels))
        ])

        self.out_conv = nnx.Conv(
            channels[-1], config.out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )

    def __call__(self, z: jax.Array):                              # (B, latent_dim)
        B = z.shape[0]
        x = self.input_proj(z)
        x = x.reshape(B, self.config.base_size, self.config.base_size, self.config.base_channels)

        for block in self.blocks:
            x = block(x)

        x = self.out_conv(x)                                       # (B, image_size, image_size, out_channels)
        return jax.nn.sigmoid(x)                                   # matches data.py's [0, 1]-normalized frames


if __name__ == "__main__":
    config = DecoderConfig(
        latent_dim=64, image_size=224, base_size=7, base_channels=256,
        stage_channels=(128, 64, 32, 16, 8),
    )
    decoder = Decoder(config, rngs=nnx.Rngs(0))
    z = jnp.zeros((4, 64))
    out = decoder(z)
    print("decoder output:", out.shape)
