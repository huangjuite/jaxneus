import jax
from jax import numpy as jnp
from functools import partial
from pyhocon import ConfigFactory

from models.fields import SDFNetwork, NeRF, RenderingNetwork, SingleVarianceNetwork
from models.renderer import extract_fields, extract_geometry, NeusRenderer

extract_fields(
    jnp.array([-1, -1, -1]),
    jnp.array([1.0, 1.0, 1.0]),
    lambda x: jnp.sum(x, axis=-1),
)

extract_geometry(
    jnp.array([-1, -1, -1]),
    jnp.array([1.0, 1.0, 1.0]),
    lambda x: jnp.sum(x, axis=-1),
)


f = open("confs/womask.conf")
conf_text = f.read()
f.close()
conf = ConfigFactory.parse_string(conf_text)

sdf_net = SDFNetwork(**conf["model.sdf_network"])
nerf = NeRF(**conf["model.nerf"])
color_net = RenderingNetwork(**conf["model.rendering_network"])
var = SingleVarianceNetwork(**conf["model.variance_network"])

renderer = NeusRenderer(nerf, sdf_net, var, color_net, **conf["model.neus_renderer"])
param = renderer.init(jax.random.PRNGKey(777), jnp.ones((1, 3)))
# x = renderer.apply(param, jnp.ones((10, 3)))
# sdf_param = {"params": param["params"]["sdf_network"]}
# x = sdf_net.apply(sdf_param, jnp.ones((10, 3)))
