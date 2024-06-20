import jax
from jaxtyping import Array, Float, PRNGKeyArray


def sample_random_points(
    key: PRNGKeyArray, num_points: int, lower: Array, upper: Array
) -> Float[Array, "b 3"]:
    return jax.random.uniform(key, (int(num_points), 3)) * (upper - lower) + lower
