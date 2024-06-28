import jax
import numpy as np
from jax import numpy as jnp

from jaxtyping import Array as JaxArray
from jaxtyping import Float
from typing import Union


class FreqEmbedder:
    """Multi Freqquency Embedder."""

    def __init__(
        self,
        input_dims: int = 3,
        frequency_num: int = 6,
        log_sampling: bool = True,
    ):
        max_freq_log2 = frequency_num - 1

        if log_sampling:
            self.freq_bands = 2.0 ** jnp.linspace(0.0, max_freq_log2, frequency_num)
        else:
            self.freq_bands = jnp.linspace(2.0**0.0, 2.0**max_freq_log2, frequency_num)

        self.output_dim = input_dims + 2 * 3 * self.freq_bands.shape[0]

    def __call__(
        self, x: Float[Union[np.ndarray, JaxArray], "3"]
    ) -> Float[Union[np.ndarray, JaxArray], "3+2*3*F"]:
        sins = jnp.sin(x[None, :] * self.freq_bands[:, None])
        coss = jnp.cos(x[None, :] * self.freq_bands[:, None])
        return jnp.concat((x, sins.reshape(-1), coss.reshape(-1)))