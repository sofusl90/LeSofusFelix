import optax
from flax import nnx
from encoder import Encoder, Projector, sigreg
from predictor import Predictor
from decoder import Decoder
import jax.numpy as jnp
import jax
import orbax.checkpoint as ocp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
from dataclasses import dataclass

@dataclass
class TrainConfig:
    adamw_lr: float
    epochs: int
    batch_size: int
    seq_len: int
    sigreg_lambda: float
    recon_lambda: float
    weight_decay: float
    grad_clip: float
    warmup_steps: int


class LeWM(nnx.Module):
    def __init__(self, enc_config, pred_config, dec_config, rngs: nnx.Rngs):
        self.encoder = Encoder(enc_config, rngs=rngs)
        self.predictor = Predictor(pred_config, rngs=rngs)
        self.pred_proj = Projector(
            pred_config.latent_dim, pred_config.proj_hidden_dim,
            pred_config.latent_dim, rngs=rngs, dtype=pred_config.dtype,
        )
        self.decoder = Decoder(dec_config, rngs=rngs)

    def encode(self, obs, chunk_size=128):             # (B, T, H, W, C)
        B, T = obs.shape[:2]
        frames = obs.reshape(B * T, *obs.shape[2:])   # (B*T, H, W, C)
        # Chunked because XLA's CPU backend aborts (SIGABRT, "is statically false")
        # differentiating this encoder's conv+layernorm+linear chain past ~170
        # frames. Remat'd because storing every chunk's activations for the
        # backward pass exhausts the 16GB card at B*T=512; recomputing them
        # costs ~1/3 extra forward time.
        enc = nnx.remat(lambda m, x: m(x))
        zs = [enc(self.encoder, frames[i : i + chunk_size]) for i in range(0, B * T, chunk_size)]
        z = jnp.concatenate(zs, axis=0)                # (B*T, D)
        return z.reshape(B, T, -1)                    # (B, T, D)

    def decode(self, emb, chunk_size=128):              # (B, T, D)
        B, T = emb.shape[:2]
        latents = emb.reshape(B * T, -1)               # (B*T, D)
        # same chunking + remat as encode(), for the same reasons.
        dec = nnx.remat(lambda m, z: m(z))
        frames = [dec(self.decoder, latents[i : i + chunk_size]) for i in range(0, B * T, chunk_size)]
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


def loss_fn(model: LeWM, obs, actions, key, sigreg_lambd, recon_lambd):
    emb = model.encode(obs).astype(jnp.float32)       # (B, T, D)
    T = emb.shape[1]

    z_hat = model.pred_proj(model.predictor(emb[:, :T-1], actions))  # actions: (B, T-1, A)
    pred_loss = jnp.mean(jnp.square(z_hat.astype(jnp.float32) - emb[:, 1:]))
    sig_loss = sigreg(emb.transpose(1, 0, 2), key)    # (T, B, D)

    # visualization-only probe: decoder must not shape the encoder's representation
    recon = model.decode(jax.lax.stop_gradient(emb)).astype(jnp.float32)  # (B, T, H, W, C)
    recon_loss = jnp.mean(jnp.square(recon - obs))

    metrics = {
        "pred": pred_loss,
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


def train(model, config, dataloader, key, run_dir):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.adamw_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.epochs * dataloader.num_batches,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_clip),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    model.train()

    ckpt_mngr = ocp.CheckpointManager(
        (run_dir / "checkpoints").resolve(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
    )
    start_epoch = 0
    if ckpt_mngr.latest_step() is not None:
        start_epoch = ckpt_mngr.latest_step() + 1
        restore_checkpoint(ckpt_mngr, ckpt_mngr.latest_step(), model, optimizer)
        print(f"resumed from checkpoint, starting at epoch {start_epoch}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Loss history lives beside the checkpoints so a resumed run extends the
    # curve instead of overwriting loss.png with a re-zeroed one.
    losses_path = run_dir / "losses.json"
    if losses_path.exists():
        saved = json.loads(losses_path.read_text())
        step_losses, epoch_losses = saved["step"], saved["epoch"]
    else:
        step_losses, epoch_losses = [], []

    for epoch in range(start_epoch, config.epochs):
        dataloader.epoch = epoch
        start = len(step_losses)
        # RNG a pure function of (base key, epoch, step), matching the dataloader,
        # so a resumed run draws the same dropout/sigreg noise as an uninterrupted one.
        epoch_key = jax.random.fold_in(key, epoch)
        for step, (obs, actions) in enumerate(dataloader):
            sk = jax.random.fold_in(epoch_key, step)
            loss, metrics, recon_sample = train_step(
                model, optimizer, obs, actions, sk, config.sigreg_lambda, config.recon_lambda
            )
            step_losses.append(float(loss))
            m = {k: float(v) for k, v in metrics.items()}
            print(
                f"step {len(step_losses) - 1:>5}  loss {float(loss):7.4f}  "
                f"pred {m['pred']:7.4f}  copy {m['copy']:7.4f}  sigreg {m['sigreg']:6.3f}  "
                f"recon {m['recon']:7.4f}  rank {m['eff_rank']:5.1f}  tvar {m['t_ratio']:.3f}"
            )
            save_loss_plot(step_losses, epoch_losses, plots_dir / "loss.png")
            save_recon_plot(*recon_sample, plots_dir / "recon.png")

        epoch_loss_slice = step_losses[start:]
        epoch_losses.append(sum(epoch_loss_slice) / len(epoch_loss_slice))
        save_checkpoint(ckpt_mngr, epoch, model, optimizer)
        ckpt_mngr.wait_until_finished()
        losses_path.write_text(json.dumps({"step": step_losses, "epoch": epoch_losses}))
        print(f"saved checkpoint at epoch {epoch}")
