
from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import nnx

# torch's nn.GELU() is the exact erf form; jax.nn.gelu defaults to the tanh
# approximation, so pin it to match the original.
exact_gelu = partial(jax.nn.gelu, approximate=False)


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class Identity(nnx.Module):
    """Stand-in for torch.nn.Identity"""

    def __call__(self, x):
        return x


class SIGReg(nnx.Module):
    def __init__(self, knots=17, num_proj=1024, *, rngs: nnx.Rngs):
        self.num_proj = num_proj
        self.rngs = rngs
        t = jnp.linspace(0.0, 3.0, knots, dtype=jnp.float32)
        dt = 3 / (knots - 1)
        weights = jnp.full((knots,), 2 * dt, dtype=jnp.float32)
        weights = weights.at[0].set(dt).at[-1].set(dt)  # torch: weights[[0, -1]] = dt
        window = jnp.exp(-jnp.square(t) / 2.0)
        # non-trainable buffers (nnx.Variable, not nnx.Param, so optimizers skip them)
        self.t = nnx.Variable(t)
        self.phi = nnx.Variable(window)
        self.weights = nnx.Variable(weights * window)

    def __call__(self, proj, key=None):
        # proj: (..., N, d)

        # sample random projections -- fresh key on every call
        if key is None:
            key = self.rngs.sigreg()
        A = jax.random.normal(key, (proj.shape[-1], self.num_proj))  # (d, num_proj)
        A = A / jnp.linalg.norm(A, axis=0)
        # compute the epps-pulley statistic
        x_t = (proj @ A)[..., None] * self.t[...]                    # (..., N, num_proj, knots)
        err = jnp.square(jnp.cos(x_t).mean(-3) - self.phi[...]) + jnp.square(
            jnp.sin(x_t).mean(-3)
        )                                                            # (..., num_proj, knots)
        statistic = (err @ self.weights[...]) * proj.shape[-2]       # (..., num_proj)
        return statistic.mean()  # average over projections and time


class FeedForward(nnx.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)  # torch default eps
        self.fc1 = nnx.Linear(dim, hidden_dim, rngs=rngs)
        self.drop1 = nnx.Dropout(dropout, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, dim, rngs=rngs)
        self.drop2 = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        x = self.norm(x)
        x = self.drop1(exact_gelu(self.fc1(x)))
        return self.drop2(self.fc2(x))


class Attention(nnx.Module):
    """Scaled dot-product attention with causal masking.

    Built on flax's nnx.MultiHeadAttention, whose internal
    nnx.dot_product_attention supports attention-weight dropout (equivalent
    to torch SDPA's dropout_p) and the standard 1/sqrt(dim_head) scaling.
    Differences vs. the torch original, forced by MHA's interface:
      * use_bias covers qkv AND the output projection, so the out proj loses
        its bias (torch had bias on out proj only)
      * the output projection is always present (torch skipped it when
        heads == 1 and dim_head == dim)
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, *, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dim, epsilon=1e-5, rngs=rngs)
        self.mha = nnx.MultiHeadAttention(
            num_heads=heads,
            in_features=dim,
            qkv_features=dim_head * heads,
            out_features=dim,
            dropout_rate=dropout,      # dropout on the attention weights
            broadcast_dropout=False,   # elementwise, like torch SDPA
            use_bias=False,            # torch original: qkv has no bias
            decode=False,
            deterministic=False,       # flipped by model.train()/.eval()
            rngs=rngs,
        )
        # torch original also has Dropout after the output projection
        self.out_drop = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, causal=True):
        x = self.norm(x)
        mask = nnx.make_causal_mask(x[..., 0]) if causal else None
        return self.out_drop(self.mha(x, mask=mask))


class ConditionalBlock(nnx.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, *, rngs: nnx.Rngs):
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rngs=rngs)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs)
        # zero init (the "-zero" in AdaLN-zero): every block starts as the identity
        self.adaLN_modulation = nnx.Linear(
            dim,
            6 * dim,
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(jax.nn.silu(c)), 6, axis=-1
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nnx.Module):

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0, *, rngs: nnx.Rngs):
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, rngs=rngs)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout, rngs=rngs)
        self.norm1 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, epsilon=1e-6, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nnx.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.input_proj = (
            nnx.Linear(input_dim, hidden_dim, rngs=rngs)
            if input_dim != hidden_dim
            else Identity()
        )
        self.cond_proj = (
            nnx.Linear(input_dim, hidden_dim, rngs=rngs)
            if input_dim != hidden_dim
            else Identity()
        )
        self.output_proj = (
            nnx.Linear(hidden_dim, output_dim, rngs=rngs)
            if hidden_dim != output_dim
            else Identity()
        )
        self.layers = nnx.List(
            block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, rngs=rngs)
            for _ in range(depth)
        )

    def __call__(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        return self.output_proj(x)


class Embedder(nnx.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
        *,
        rngs: nnx.Rngs,
    ):
        # flax convs are channels-last (B, T, C), so the permutes are gone
        self.patch_embed = nnx.Conv(input_dim, smoothed_dim, kernel_size=(1,), strides=1, rngs=rngs)
        self.fc1 = nnx.Linear(smoothed_dim, mlp_scale * emb_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_scale * emb_dim, emb_dim, rngs=rngs)

    def __call__(self, x):

        x = x.astype(jnp.float32)
        x = self.patch_embed(x)
        x = self.fc2(jax.nn.silu(self.fc1(x)))
        return x


class MLP(nnx.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nnx.LayerNorm,
        act_fn=exact_gelu,
        *,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.norm = norm_fn(hidden_dim, rngs=rngs) if norm_fn is not None else Identity()
        self.act = act_fn
        self.fc2 = nnx.Linear(hidden_dim, output_dim or input_dim, rngs=rngs)

    def __call__(self, x):

        return self.fc2(self.act(self.norm(self.fc1(x))))


class ARPredictor(nnx.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        rngs: nnx.Rngs,
    ):
        self.pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_frames, input_dim))
        )
        self.dropout = nnx.Dropout(emb_dropout, rngs=rngs)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            rngs=rngs,
        )

    def __call__(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.shape[1]
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


if __name__ == "__main__":
    import optax

    rngs = nnx.Rngs(0)  # one seed; params/dropout streams fork from it
    model = ARPredictor(
        num_frames=16, depth=2, heads=4, mlp_dim=256,
        input_dim=32, hidden_dim=128, dim_head=32,
        dropout=0.1, emb_dropout=0.1, rngs=rngs,
    )
    sigreg = SIGReg(rngs=nnx.Rngs(1))  # its own stream

    x = jax.random.normal(jax.random.key(2), (8, 16, 32))
    c = jax.random.normal(jax.random.key(3), (8, 16, 32))

    model.eval()
    print("eval output:", model(x, c).shape)

    model.train()
    optimizer = nnx.Optimizer(model, optax.adamw(3e-4), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, sigreg, optimizer, x, c):
        # modules with mutable state (dropout rngs, sigreg's projection rng) must
        # be *arguments* of the grad-transformed function, not closed-over
        def loss_fn(model, sigreg):
            pred = model(x, c)
            reg = sigreg(rearrange(pred, "b t d -> t b d"))
            return jnp.square(pred[:, :-1] - x[:, 1:]).mean() + 0.01 * reg

        # argnums defaults to 0 -> grads w.r.t. model params only
        loss, grads = nnx.value_and_grad(loss_fn)(model, sigreg)
        optimizer.update(model, grads)  # flax < 0.11: optimizer.update(grads)
        return loss

    for step in range(3):
        print("step", step, "loss:", float(train_step(model, sigreg, optimizer, x, c)))
