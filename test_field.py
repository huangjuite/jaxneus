import os
import numpy as np
import jax
import optax
from jax import numpy as jnp
from tqdm import trange
from functools import partial
import trimesh
import time

from models.fields import SDFNetwork, RenderingNetwork, NeRF, SingleVarianceNetwork


from pyhocon import ConfigFactory

f = open("confs/womask.conf")
conf_text = f.read()
f.close()
conf = ConfigFactory.parse_string(conf_text)

key = jax.random.PRNGKey(777)

sdf = SDFNetwork(**conf["model.sdf_network"])
params = sdf.init(key, jnp.zeros((1, 3)))
d, feat = sdf.apply(params, jnp.zeros((10, 3)))
print(d.shape, feat.shape)

nerf = NeRF(**conf["model.nerf"])
params = nerf.init(key, jnp.ones((1, 4)), jnp.ones((1, 4)))
alpha, rgb = nerf.apply(params, jnp.ones((10, 4)), jnp.ones((10, 4)))
print(alpha.shape, rgb.shape)

rend = RenderingNetwork(**conf["model.rendering_network"])
params = rend.init(
    key, jnp.ones((1, 3)), jnp.ones((1, 3)), jnp.ones((1, 3)), jnp.ones((1, 256))
)
rgb = rend.apply(
    params, jnp.ones((10, 3)), jnp.ones((10, 3)), jnp.ones((10, 3)), jnp.ones((10, 256))
)
print(rgb.shape)


var = SingleVarianceNetwork(**conf["model.variance_network"])
params = var.init(key, jnp.ones((1, 3)))
variance = var.apply(params, jnp.ones((10, 3)))
print(variance.shape)
