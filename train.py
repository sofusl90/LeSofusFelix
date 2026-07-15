import optax
from flax import nnx
from encoder import Encoder, sigreg
from predictor import Predictor
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class TrainConfig:
    adamw_lr: float
    epochs: int


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
    B, T, _ = emb.shape

    state = jnp.broadcast_to(
        model.init_state.value, (B, *model.init_state.value.shape[1:])
    )                                                 # (B, Ns, state_dim)

    preds = []
    for t in range(T - 1):
        z_t = emb[:, t][:, None, :]                   # (B, 1, D)
        z_hat, state = model.predictor(z_t, state, actions[:, t])
        preds.append(z_hat[:, 0])                     # (B, D)
    preds = jnp.stack(preds, axis=1)                  # (B, T-1, D)

    pred_loss = jnp.mean(jnp.square(preds - emb[:, 1:]))
    sig_loss = sigreg(emb.transpose(1, 0, 2), key)    # (T, B, D)
    return pred_loss + lambd * sig_loss, {"pred": pred_loss, "sigreg": sig_loss}

@nnx.jit
def train_step(model, optimizer, obs, actions, key):
    model.train()
    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, obs, actions, key
    )
    optimizer.update(grads)
    return loss, metrics


def train(model, config, dataloader, key):
    optimizer = optax.adamw(config.adamw_lr)

    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    losses = []
    plt.ion()
    plt.show()

    for _ in range(config.epochs):
        epoch_losses = []
        for obs, actions in dataloader:
            key, sk = jax.random.split(key)
            loss, metrics = train_step(model, optimizer, obs, actions, sk)
            epoch_losses.append(loss)

        losses.append(sum(epoch_losses) / len(epoch_losses))
        line.set_data(range(len(losses)), losses)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
