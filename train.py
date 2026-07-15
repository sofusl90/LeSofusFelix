import optax
from flax import nnx
from encoder import Encoder, sigreg
from predictor import Predictor
import jax.numpy as jnp
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class TrainConfig:
    adamw_lr: float
    epochs: int
    batch_size: int
    seq_len: int


class LeWM(nnx.Module):
    def __init__(self, enc_config, pred_config, rngs: nnx.Rngs):
        self.encoder = Encoder(enc_config, rngs=rngs)
        self.predictor = Predictor(pred_config, rngs=rngs)
        self.init_state = nnx.Param(
            jnp.zeros((1, pred_config.num_state_tokens, pred_config.state_dim))
        )

    def encode(self, obs):                            # (B, T, H, W, C)
        B, T = obs.shape[:2]
        frames = obs.reshape(B * T, *obs.shape[2:])   # (B*T, H, W, C)
        z = self.encoder(frames)                      # (B*T, D)
        return z.reshape(B, T, -1)                    # (B, T, D)




def loss_fn(model: LeWM, obs, actions, key, lambd=0.1):
    emb = model.encode(obs)                      # (B, T, D)
    B, T, D = emb.shape

    state = jnp.broadcast_to(
        model.init_state.value, (B, *model.init_state.value.shape[1:])
    )                                                 # (B, Ns, state_dim)
    z_state = jnp.zeros((B, model.predictor.config.state_dim))

    preds = []
    for t in range(T - 1):
        z_t = jnp.concatenate([emb[:, t], z_state], axis=-1)[:, None, :]  # (B, 1, D + state_dim)
        z_hat, state = model.predictor(z_t, state, actions[:, t])
        preds.append(z_hat[:, 0, :D])                 # (B, D)
        z_state = z_hat[:, 0, D:]                     # (B, state_dim)
    preds = jnp.stack(preds, axis=1)                  # (B, T-1, D)

    pred_loss = jnp.mean(jnp.square(preds - emb[:, 1:]))
    sig_loss = sigreg(emb.transpose(1, 0, 2), key)    # (T, B, D)
    return pred_loss + lambd * sig_loss, {"pred": pred_loss, "sigreg": sig_loss}


@nnx.jit
def train_step(model, optimizer, obs, actions, key):
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, obs, actions, key
    )
    optimizer.update(model, grads)
    return loss, metrics


def save_loss_plot(step_losses, epoch_losses, path):
    fig, (ax_step, ax_epoch) = plt.subplots(1, 2, figsize=(10, 4))
    ax_step.plot(step_losses)
    ax_step.set_xlabel("step")
    ax_step.set_ylabel("loss")
    ax_epoch.plot(epoch_losses)
    ax_epoch.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(model, config, dataloader, key):
    optimizer = nnx.Optimizer(model, optax.adamw(config.adamw_lr), wrt=nnx.Param)
    model.train()

    run_dir = Path("plots") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    step_losses, epoch_losses = [], []
    for _ in range(config.epochs):
        start = len(step_losses)
        for obs, actions in dataloader:
            key, sk = jax.random.split(key)
            loss, metrics = train_step(model, optimizer, obs, actions, sk)
            step_losses.append(float(loss))
            save_loss_plot(step_losses, epoch_losses, run_dir / "loss.png")

        epoch = step_losses[start:]
        epoch_losses.append(sum(epoch) / len(epoch))
