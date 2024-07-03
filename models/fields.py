import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from flax import linen as nn
from models.embedder import FreqEmbedder


def InitNormal(mean=0.0, stddev=1.0):
    def init(key, shape, dtype=jnp.float32):
        return mean + stddev * jax.random.normal(key, shape, dtype)

    return init


class SeperateInitializedDense(nn.Module):
    """variance scaling with scale 2.0, fan_out, and normal for input 1,
    zeros for input 2
    """

    features: int

    @nn.compact
    def __call__(
        self, x1: Float[Array, "b f1"], x2: Float[Array, "b f2"]
    ) -> Float[Array, "b f"]:
        w1 = self.param(
            "kernel/w1",
            InitNormal(stddev=(jnp.sqrt(2.0) / jnp.sqrt(self.features))),
            (x1.shape[-1], self.features),
        )
        w2 = self.param(
            "kernel/w2", nn.initializers.zeros_init(), (x2.shape[-1], self.features)
        )
        bias = self.param("bias", nn.initializers.zeros_init(), (self.features,))
        return jnp.dot(x1, w1) + jnp.dot(x2, w2) + bias


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
            self.embed_fn_fine = FreqEmbedder(frequency_num=self.multires)
            dims[0] = self.embed_fn_fine.output_dim

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if self.geometric_init:
                if l == self.num_layers - 2:
                    mean = jnp.sqrt(jnp.pi) / jnp.sqrt(dims[l])
                    mean *= -1 if self.inside_outside else 1
                    bias = self.bias if self.inside_outside else -self.bias
                    lin = nn.Dense(
                        out_dim,
                        kernel_init=InitNormal(mean=mean, stddev=0.0001),
                        bias_init=nn.initializers.constant(bias / jnp.pi),
                    )
                elif self.multires > 0 and (l == 0 or l in self.skip_in):
                    lin = SeperateInitializedDense(out_dim)
                else:
                    lin = nn.Dense(
                        out_dim,
                        kernel_init=InitNormal(
                            stddev=(jnp.sqrt(2.0) / jnp.sqrt(out_dim))
                        ),
                        bias_init=nn.initializers.zeros_init(),
                    )

            if self.weight_norm:
                lin = nn.WeightNorm(lin, variable_filter={"kernel"}, use_scale=False)

            setattr(self, f"lin_{l}", lin)

    def _softplus(self, x, beta=1, threshold=20):
        x_safe = jax.lax.select(x * beta < threshold, x, jnp.ones_like(x))
        return jax.lax.select(
            x * beta < threshold, 1 / beta * jnp.log(1 + jnp.exp(beta * x_safe)), x
        )

    def __call__(
        self, inputs: Float[Array, "b 3"]
    ) -> tuple[Float[Array, "b 1"], Float[Array, "b f"]]:
        inputs = inputs * self.scale
        if self.embed_fn_fine:
            inputs = jax.vmap(self.embed_fn_fine)(inputs)
        x = inputs
        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin_{l}")

            if l == 0 and self.multires > 0:
                x = lin(x[:, :3], x[:, 3:])
            elif l in self.skip_in and self.multires > 0:
                x = lin(
                    jnp.concatenate([x, inputs[:, :3]], axis=-1) / jnp.sqrt(2),
                    inputs[:, 3:] / jnp.sqrt(2),
                )
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                x = self._softplus(x, beta=100)

        sd = x[:, :1] / self.scale
        feature = x[:, 1:]
        return jnp.concatenate([sd, feature], axis=-1)


class RenderingNetwork(nn.Module):
    d_feature: int
    mode: str
    d_in: int
    d_out: int
    d_hidden: int
    n_layers: int
    weight_norm: bool = True
    multires_view: int = 0
    squeeze_out: bool = True

    def setup(self):
        dims = (
            [self.d_in + self.d_feature]
            + [self.d_hidden] * self.n_layers
            + [self.d_out]
        )
        self.embed_fn_view = None
        if self.multires_view > 0:
            self.embed_fn_view = FreqEmbedder(frequency_num=self.multires_view)
            dims[0] += self.embed_fn_view.output_dim - 3

        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            lin = nn.Dense(dims[l + 1])
            if self.weight_norm:
                lin = nn.WeightNorm(lin)
            setattr(self, f"lin_{l}", lin)

    def __call__(
        self,
        points: Float[Array, "b 3"],
        normals: Float[Array, "b 3"],
        view_dirs: Float[Array, "b 3"],
        features: Float[Array, "b f"],
    ) -> Float[Array, "b 3"]:

        if self.embed_fn_view:
            view_dirs = jax.vmap(self.embed_fn_view)(view_dirs)

        rendering_input = None
        if self.mode == "idr":
            rendering_input = jnp.concatenate(
                [points, normals, view_dirs, features], axis=-1
            )
        elif self.mode == "no_view_dir":
            rendering_input = jnp.concatenate([points, normals, features], axis=-1)
        elif self.mode == "no_normal":
            rendering_input = jnp.concatenate([points, view_dirs, features], axis=-1)
        x = rendering_input

        for l in range(self.num_layers - 1):
            lin = getattr(self, f"lin_{l}")
            x = lin(x)
            if l < self.num_layers - 2:
                x = nn.relu(x)

        if self.squeeze_out:
            x = nn.sigmoid(x)

        return x


class NeRF(nn.Module):
    """NeRF Model."""

    D: int = 8
    W: int = 256
    d_in: int = 3
    d_in_view: int = 3
    multires: int = 0
    multires_view: int = 0
    output_ch: int = 4
    skips: tuple = (4,)
    use_viewdirs: bool = False

    def setup(self):
        self.input_ch, self.input_ch_view = self.d_in, self.d_in_view
        self.embed_fn, self.embed_fn_view = None, None
        if self.multires > 0:
            self.embed_fn = FreqEmbedder(
                input_dims=self.d_in, frequency_num=self.multires
            )
            self.input_ch = self.embed_fn.output_dim

        if self.multires_view > 0:
            self.embed_fn_view = FreqEmbedder(
                input_dims=self.d_in_view, frequency_num=self.multires_view
            )
            self.input_ch_view = self.embed_fn_view.output_dim

        self.pts_linears = [nn.Dense(self.W) for _ in range(self.D)]
        self.view_linears = [nn.Dense(self.W // 2)]

        if self.use_viewdirs:
            self.feature_linear = nn.Dense(self.W)
            self.alpha_linear = nn.Dense(1)
            self.rgb_linear = nn.Dense(3)
        else:
            self.output_linear = nn.Dense(self.output_ch)

    def __call__(
        self, input_pts: Float[Array, "b d_in"], input_views: Float[Array, "b d_in"]
    ) -> tuple[Float[Array, "b 1"], Float[Array, "b 3"]]:
        if self.embed_fn:
            input_pts = jax.vmap(self.embed_fn)(input_pts)
        if self.embed_fn_view:
            input_views = jax.vmap(self.embed_fn_view)(input_views)

        h = input_pts
        for i in range(len(self.pts_linears)):
            h = nn.relu(self.pts_linears[i](h))
            if i in self.skips:
                h = jnp.concatenate([input_pts, h], axis=-1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jnp.concatenate([feature, input_views], axis=-1)

            for i in range(len(self.view_linears)):
                h = nn.relu(self.view_linears[i](h))

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False, "Not implemented"


class SingleVarianceNetwork(nn.Module):
    init_val: float

    def setup(self):
        self.variance = self.init_val * self.param("variance", nn.initializers.ones, ())

    def __call__(self, x: Float[Array, "b"]) -> Float[Array, "b"]:
        return jnp.ones((x.shape[0])) * jnp.exp(self.variance * 10)
