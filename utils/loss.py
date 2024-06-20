from jax import numpy as jnp
from jaxtyping import Float, Array


def eikonal_loss(gradients: Float[Array, "b 3"]):
    gradient_error = (jnp.linalg.norm(gradients, ord=2, axis=-1) - 1.0) ** 2
    return jnp.mean(gradient_error)

def random_pts_sdf_loss(signed_distance: Float[Array, "b 1"]):
    return jnp.exp(-1e2 * jnp.abs(signed_distance)).mean()