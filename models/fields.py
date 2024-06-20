import jax
from jax import numpy as jnp

from jaxtyping import Array, Float
from typing import ClassVar

from models.embedder import FreqEmbedder
from flax import linen as nn


class SDFNetwork(nn.Module):
    d_in: int
    d_out: int
    d_hidden: int
    n_layers: int
    skip_in: tuple = (4,)
    multires: int = 10
    bias: float = 0.5
    scale: float = 1.0
    geometric_init: bool = True
    weight_norm: bool = True
    inside_outside: bool = False

    def setup(self):
        dims = [self.d_in] + [self.d_hidden] * self.n_layers + [self.d_out]
        self.embed_fn_fine = None
        if self.multires > 0:
            self.embed_fn_fine = FreqEmbedder()
            dims[0] = self.embed_fn_fine.output_dim

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Dense(
                out_dim,
                kernel_init=nn.initializers.normal(stddev=jnp.sqrt(2 / out_dim)),
            )

            if self.weight_norm:
                lin = nn.WeightNorm(lin)

            setattr(self, f"lin_{l}", lin)

    def __call__(self, inputs: Float[Array, "b 3"]) -> Float[Array, "b 257"]:
        inputs = inputs * self.scale
        if self.embed_fn_fine:
            inputs = jax.vmap(self.embed_fn_fine)(inputs)
        x = inputs
        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin_{l}")

            if l in self.skip_in:
                x = jnp.concatenate([inputs, x], axis=-1)

            x = lin(x)
            if l < self.num_layers - 2:
                x = nn.softplus(x)

        x = jnp.concat([x[:, :1] / self.scale, x[:, 1:]], axis=-1)
        return x
