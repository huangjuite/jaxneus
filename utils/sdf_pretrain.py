import jax
import optax
from jax import numpy as jnp
from jaxtyping import PyTree, Float, Array, Int
from functools import partial
from tqdm import trange

from models.fields import SDFNetwork


def sample_random_points(
    key, num_points: int, lower: Array, upper: Array
) -> Float[Array, "b 3"]:
    return jax.random.uniform(key, (int(num_points), 3)) * (upper - lower) + lower


def sdf_pretrain(
    net: SDFNetwork,
    key=jax.random.PRNGKey(777),
    bbox_min=jnp.array([-1.01, -1.01, -1.01]),
    bbox_max=jnp.array([1.01, 1.01, 1.01]),
    iteration=2000,
    batch_size=2**15,
    sphere_radius=0.5,
):
    params = net.init(key, jnp.zeros((1, 3)))
    optimizer = optax.adam(learning_rate=5e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(param: PyTree, points: Float[Array, "b 3"], gt_sd: Float[Array, "b 3"]):
        sdf = partial(net.apply, param)
        sd = sdf(points)[:, 0]
        loss = jnp.mean((sd - gt_sd) ** 2)
        sdf_func = lambda x: jnp.sum(sdf(x)[:, 0])
        gradients = jax.vmap(jax.grad(sdf_func))(points[:, None, :])
        grad_loss = jnp.mean((jnp.linalg.norm(gradients, ord=2, axis=-1) - 1.0) ** 2)
        return loss + 0.1 * grad_loss

    @jax.jit
    def step_once(rng, param, opt_state):
        points = sample_random_points(rng, batch_size / 10, bbox_min, bbox_max)
        gt_sd = jnp.linalg.norm(points, axis=-1) - sphere_radius

        loss, grads = jax.value_and_grad(loss_fn)(param, points, gt_sd)
        updates, opt_state = optimizer.update(grads, opt_state)
        param = optax.apply_updates(param, updates)

        return loss, param, opt_state

    pbar = trange(iteration)
    for i in pbar:
        key, rng = jax.random.split(key, 2)
        loss, params, opt_state = step_once(rng, params, opt_state)
        pbar.set_description("Iteration %d, Loss: %.5f" % (i, loss))

    return params
