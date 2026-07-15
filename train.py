
import optax
from flax import nnx
from encoder import Encoder, sigreg
from predictor import Predictor


class LeWM(nnx.Module):
    def __init__(self, enc_config, pred_config, rngs: nnx.Rngs):
        self.encoder = Encoder(enc_config, rngs=rngs)
        self.predictor = Predictor(pred_config, rngs=rngs)

    def encode(self, obs):
        B, T = obs.shape[:2]
        frames = obs.reshape(B * T, *obs.shape[2:])
        z = self.encoder(frames)
        return z.reshape(B, T, -1)



# key, sk = jax.random.split(key)
# emb = encoder(obs)                                  # (B, T, D)
# loss = pred_loss + lambd * sigreg(emb.transpose(1, 0, 2), sk)   # (T, B, D)
