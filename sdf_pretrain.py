import os
import numpy as np
import jax
import optax
from jax import numpy as jnp
from tqdm import trange
from functools import partial
import trimesh

from models.fields import SDFNetwork

from utils.sample import sample_random_points
from utils.visualize import extract_geometry
from utils.loss import eikonal_loss

key = jax.random.PRNGKey(777)
base_exp_dir = "exp/sdf_pretrain"
os.makedirs(base_exp_dir, exist_ok=True)

net = SDFNetwork(d_in=3, d_out=257, d_hidden=256, n_layers=8)
params = net.init(key, jnp.zeros((1, 3)))
optimizer = optax.adam(learning_rate=5e-3)
opt_state = optimizer.init(params)

iteration = 1500
batch_size = 2**15
bbox_min = jnp.array([-1.01, -1.01, -1.01])
bbox_max = jnp.array([1.01, 1.01, 1.01])
resolution = 128


@jax.jit
def loss_fn(params, points, gt_sd):

    sdf = partial(net.apply, params)

    sd = sdf(points)[:, 0]
    loss = jnp.mean((sd - gt_sd) ** 2)

    def sdf_single(points):
        return jnp.sum(sdf(points)[:, 0])

    gradients = jax.vmap(jax.grad(sdf_single))(points[:, None, :])
    grad_loss = eikonal_loss(gradients)

    return loss + 0.1 * grad_loss


pbar = trange(iteration + 1)
for i in pbar:
    key, rng = jax.random.split(key, 2)
    points = sample_random_points(rng, batch_size / 10, bbox_min, bbox_max)
    gt_sd = jnp.linalg.norm(points, axis=-1) - 0.5

    loss, grads = jax.value_and_grad(loss_fn)(params, points, gt_sd)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    pbar.set_description("Iteration %d, Loss: %.5f" % (i, loss))


vertices, triangles = extract_geometry(
    bbox_min,
    bbox_max,
    resolution,
    threshold=0.0,
    query_func=lambda x: -net.apply(params, x)[:, 0],
    batch_size=batch_size,
)
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
vertices = vertices @ R
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export(os.path.join(base_exp_dir, "mesh.glb"))
