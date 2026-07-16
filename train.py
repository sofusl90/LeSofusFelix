import optax
from flax import nnx
from encoder import Encoder, sigreg
from predictor import Predictor
from decoder import Decoder
import jax.numpy as jnp
import jax
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CHECKPOINT_DIR = Path("checkpoints").resolve()

@dataclass
class TrainConfig:
    adamw_lr: float
    epochs: int
    batch_size: int
    seq_len: int
    sigreg_lambda: float
    recon_lambda: float


class LeWM(nnx.Module):
    def __init__(self, enc_config, pred_config, dec_config, rngs: nnx.Rngs):
        self.encoder = Encoder(enc_config, rngs=rngs)
        self.predictor = Predictor(pred_config, rngs=rngs)
        self.decoder = Decoder(dec_config, rngs=rngs)
        self.init_state = nnx.Param(
            jnp.zeros((1, pred_config.num_state_tokens, pred_config.state_dim))
        )

    def encode(self, obs, chunk_size=128):             # (B, T, H, W, C)
        B, T = obs.shape[:2]
        frames = obs.reshape(B * T, *obs.shape[2:])   # (B*T, H, W, C)
        # XLA's CPU backend aborts (SIGABRT, "is statically false") differentiating
        # this encoder's conv+layernorm+linear chain once the batch exceeds ~170 on
        # this machine, so the flattened B*T batch is run through in chunks.
        zs = [self.encoder(frames[i : i + chunk_size]) for i in range(0, B * T, chunk_size)]
        z = jnp.concatenate(zs, axis=0)                # (B*T, D)
        return z.reshape(B, T, -1)                    # (B, T, D)

    def decode(self, emb, chunk_size=128):              # (B, T, D)
        B, T = emb.shape[:2]
        latents = emb.reshape(B * T, -1)               # (B*T, D)
        # same chunking as encode(), for the same XLA CPU backend reason.
        frames = [self.decoder(latents[i : i + chunk_size]) for i in range(0, B * T, chunk_size)]
        frames = jnp.concatenate(frames, axis=0)        # (B*T, H, W, C)
        return frames.reshape(B, T, *frames.shape[1:])  # (B, T, H, W, C)




def collapse_stats(emb):
    """Healthy: eff_rank in the tens (of D), t_ratio well above 0, copy > pred."""
    emb = jax.lax.stop_gradient(emb)
    B, T, D = emb.shape
    flat = emb.reshape(B * T, D)
    c = flat - flat.mean(0)
    eig = jnp.linalg.eigvalsh(c.T @ c / (B * T))
    return {
        "copy": jnp.mean(jnp.square(emb[:, 1:] - emb[:, :-1])),
        "eff_rank": jnp.square(eig.sum()) / jnp.square(eig).sum(),
        "t_ratio": jnp.var(emb, axis=1).mean() / jnp.var(flat, axis=0).mean(),
    }


def rollout_predictor(model: LeWM, emb, actions, carry_state: bool):
    """Predict emb[:, 1:] from emb[:, :-1]. If carry_state, the persistent multi-token
    `state` accumulates across steps as usual; if not, it's reset to init_state every
    step, ablating the persistent memory so only the single-vector z_state (LeWorldModel's
    own carry) is available -- this isolates what the added persistent state contributes.
    """
    B, T, D = emb.shape
    init_state = jnp.broadcast_to(
        model.init_state.value, (B, *model.init_state.value.shape[1:])
    )                                                 # (B, Ns, state_dim)
    state = init_state
    z_state = jnp.zeros((B, model.predictor.config.state_dim))

    preds = []
    for t in range(T - 1):
        z_t = jnp.concatenate([emb[:, t], z_state], axis=-1)[:, None, :]  # (B, 1, D + state_dim)
        z_hat, new_state = model.predictor(z_t, state, actions[:, t])
        state = new_state if carry_state else init_state
        preds.append(z_hat[:, 0, :D])                 # (B, D)
        z_state = z_hat[:, 0, D:]                     # (B, state_dim)
    return jnp.stack(preds, axis=1)                   # (B, T-1, D)


def loss_fn(model: LeWM, obs, actions, key, sigreg_lambd, recon_lambd):
    emb = model.encode(obs)                      # (B, T, D)

    preds = rollout_predictor(model, emb, actions, carry_state=True)
    pred_loss = jnp.mean(jnp.square(preds - emb[:, 1:]))

    # diagnostic only: does carrying the persistent state actually help prediction?
    # (not part of `loss`, so this never affects training -- purely observational)
    frozen_emb = jax.lax.stop_gradient(emb)
    no_state_preds = rollout_predictor(model, frozen_emb, actions, carry_state=False)
    no_state_pred_loss = jnp.mean(jnp.square(no_state_preds - frozen_emb[:, 1:]))

    sig_loss = sigreg(emb.transpose(1, 0, 2), key)    # (T, B, D)

    # visualization-only probe: decoder must not shape the encoder's representation
    recon = model.decode(jax.lax.stop_gradient(emb))  # (B, T, H, W, C)
    recon_loss = jnp.mean(jnp.square(recon - obs))

    metrics = {
        "pred": pred_loss,
        "pred_no_state": no_state_pred_loss,
        "state_gain": no_state_pred_loss - pred_loss,  # positive => state is helping
        "sigreg": sig_loss,
        "recon": recon_loss,
        **collapse_stats(emb),
    }
    loss = pred_loss + sigreg_lambd * sig_loss + recon_lambd * recon_loss
    # one sequence's worth of frames, for visualizing reconstruction quality
    recon_sample = (obs[0], recon[0])
    return loss, (metrics, recon_sample)


@nnx.jit
def train_step(model, optimizer, obs, actions, key, sigreg_lambd, recon_lambd):
    (loss, (metrics, recon_sample)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        model, obs, actions, key, sigreg_lambd, recon_lambd
    )
    optimizer.update(model, grads)
    return loss, metrics, recon_sample


def save_recon_plot(obs_seq, recon_seq, path, n=6):
    """obs_seq, recon_seq: (T, H, W, C) arrays in [0, 1] for a single sequence."""
    obs_seq, recon_seq = jax.device_get(obs_seq), jax.device_get(recon_seq)
    T = obs_seq.shape[0]
    idx = [round(i * (T - 1) / (n - 1)) for i in range(min(n, T))]

    fig, axes = plt.subplots(2, len(idx), figsize=(2 * len(idx), 4.4))
    for col, t in enumerate(idx):
        axes[0, col].imshow(obs_seq[t].clip(0, 1))
        axes[0, col].set_title(f"t={t}", fontsize=8)
        axes[1, col].imshow(recon_seq[t].clip(0, 1))
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("recon")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(mngr, step, model, optimizer):
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    mngr.save(step, args=ocp.args.Composite(
        model=ocp.args.StandardSave(model_state),
        optimizer=ocp.args.StandardSave(opt_state),
    ))


def restore_checkpoint(mngr, step, model, optimizer):
    _, abs_model_state = nnx.split(model)
    _, abs_opt_state = nnx.split(optimizer)
    restored = mngr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(abs_model_state),
        optimizer=ocp.args.StandardRestore(abs_opt_state),
    ))
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["optimizer"])


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

    ckpt_mngr = ocp.CheckpointManager(
        CHECKPOINT_DIR, options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    )
    start_epoch = 0
    if ckpt_mngr.latest_step() is not None:
        start_epoch = ckpt_mngr.latest_step() + 1
        restore_checkpoint(ckpt_mngr, ckpt_mngr.latest_step(), model, optimizer)
        print(f"resumed from checkpoint, starting at epoch {start_epoch}")

    run_dir = Path("plots") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    step_losses, epoch_losses = [], []
    for epoch in range(start_epoch, config.epochs):
        start = len(step_losses)
        for obs, actions in dataloader:
            key, sk = jax.random.split(key)
            loss, metrics, recon_sample = train_step(
                model, optimizer, obs, actions, sk, config.sigreg_lambda, config.recon_lambda
            )
            step_losses.append(float(loss))
            m = {k: float(v) for k, v in metrics.items()}
            print(
                f"step {len(step_losses) - 1:>5}  loss {float(loss):7.4f}  "
                f"pred {m['pred']:7.4f}  no_state {m['pred_no_state']:7.4f}  "
                f"state_gain {m['state_gain']:+7.4f}  copy {m['copy']:7.4f}  sigreg {m['sigreg']:6.3f}  "
                f"recon {m['recon']:7.4f}  rank {m['eff_rank']:5.1f}  tvar {m['t_ratio']:.3f}"
            )
            save_loss_plot(step_losses, epoch_losses, run_dir / "loss.png")
            save_recon_plot(*recon_sample, run_dir / "recon.png")

        epoch_loss_slice = step_losses[start:]
        epoch_losses.append(sum(epoch_loss_slice) / len(epoch_loss_slice))
        save_checkpoint(ckpt_mngr, epoch, model, optimizer)
        ckpt_mngr.wait_until_finished()
        print(f"saved checkpoint at epoch {epoch}")
