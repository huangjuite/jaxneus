import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, PyTree
import flax.linen as nn
from functools import partial

from models.fields import SDFNetwork, RenderingNetwork, NeRF, SingleVarianceNetwork


class NeusRenderer(nn.Module):
    nerf: NeRF
    sdf_network: SDFNetwork
    deviation_network: SingleVarianceNetwork
    color_network: RenderingNetwork
    n_samples: int
    n_importance: int
    n_outside: int
    up_sample_steps: int
    perturb: float

    def sample_pdf(
        self,
        bins: Float[Array, "n"],
        weights: Float[Array, "n-1"],
        n_samples: Int,
    ) -> Float[Array, "n_samples"]:
        """sample with pdf along a single ray."""
        # get pdf
        weights += 1e-5
        pdf = weights / jnp.sum(weights)
        cdf = jnp.cumsum(pdf)
        cdf = jnp.concatenate([jnp.zeros(1), cdf])

        # take uniform sammples
        u = jnp.linspace(0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, n_samples)

        # invert cdf
        inds = jnp.searchsorted(cdf, u)
        below = jnp.maximum(0, inds - 1)
        above = jnp.minimum(cdf.shape[-1] - 1, inds)
        inds_g = jnp.stack([below, above], axis=-1)

        matched_shape = [inds_g.shape[0], cdf.shape[-1]]
        cdf_g = jnp.take_along_axis(
            jnp.broadcast_to(cdf, matched_shape), inds_g, axis=-1
        )
        bins_g = jnp.take_along_axis(
            jnp.broadcast_to(bins, matched_shape), inds_g, axis=-1
        )

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = jnp.where(denom < 1e-5, 1.0, denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        return samples

    def up_sample(
        self,
        ray_o: Float[Array, "3"],
        ray_d: Float[Array, "3"],
        z_vals: Float[Array, "n"],
        sdf: Float[Array, "n"],
        n_importance: Int,
        inv_s: Int,
    ) -> Float[Array, "n_importance"]:
        """Up sampling given a fixed inv_s for a single ray."""
        pts = ray_o + ray_d * z_vals[:, None]
        radius = jnp.linalg.norm(pts, ord=2, axis=-1)
        inside_sphere = (radius[:-1] < 1.0) | (radius[1:] < 1.0)

        prev_sdf, next_sdf = sdf[:-1], sdf[1:]
        prev_z_vals, next_z_vals = z_vals[:-1], z_vals[1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5

        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        prev_cos_val = jnp.concatenate([jnp.zeros([1]), cos_val[:-1]])
        cos_val = jnp.minimum(cos_val, prev_cos_val)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = next_z_vals - prev_z_vals
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = nn.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = nn.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = (
            alpha * jnp.cumprod(jnp.concatenate([jnp.ones(1), 1.0 - alpha + 1e-7]))[:-1]
        )

        z_samples = self.sample_pdf(z_vals, weights, n_importance)
        return jax.lax.stop_gradient(z_samples)

    def cat_z_vals(
        self,
        ray_o: Float[Array, "3"],
        ray_d: Float[Array, "3"],
        z_vals: Float[Array, "n_sample"],
        new_z_vals: Float[Array, "n_new_sample"],
        sdf: Float[Array, "n_sample"],
        last: bool = False,
    ) -> tuple[
        Float[Array, "n_sample + n_new_sample"], Float[Array, "n_sample + n_new_sample"]
    ]:
        """Concatenate z_vals, sdf with new_z_val for a single ray."""
        pts = ray_o + ray_d * new_z_vals[:, None]
        z_vals = jnp.concatenate([z_vals, new_z_vals])
        index = jnp.argsort(z_vals)
        z_vals = z_vals[index]

        if not last:
            new_sdf = self.sdf_network(jax.lax.stop_gradient(pts.reshape(-1, 3)))[:, 0]
            sdf = jnp.concatenate([sdf, new_sdf.reshape(-1)])
            sdf = sdf[index]

        return z_vals, sdf

    def render_core_outside(
        self,
        ray_o: Float[Array, "3"],
        ray_d: Float[Array, "3"],
        z_vals: Float[Array, "n_sample"],
        sample_dist: Float,
        background_rgb: Float[Array, "3"] = None,
    ) -> tuple[
        Float[Array, "3"],  # color
        Float[Array, "n_sample 3"],  # sampled color
        Float[Array, "n_sample"],  # alpha
        Float[Array, "n_sample"],  # weights
    ]:
        """Render background color for single ray.

        Ooutputs:
            color: [3]
            sampled_color: [n_sample ,3]
            alpha: [n_sample]
            weights: [n_sample]
        """
        n_sample = z_vals.shape[0]

        dist = z_vals[1:] - z_vals[:-1]
        dists = jnp.concatenate([dist, jnp.array([sample_dist])])
        mid_z_vals = z_vals + dists * 0.5

        pts = ray_o + ray_d * mid_z_vals[:, None]
        dis_to_center = jnp.linalg.norm(pts, axis=1, keepdims=True).clip(1.0, 1e10)
        pts = jnp.concatenate([pts / dis_to_center, 1.0 / dis_to_center], axis=-1)
        dirs = ray_d[None, :].repeat(n_sample, axis=0)

        density, sampled_color = self.nerf(pts, dirs)
        sampled_color = nn.sigmoid(sampled_color)
        alpha = 1.0 - jnp.exp(-nn.softplus(density.squeeze()) * dists)
        trans = jnp.cumprod(jnp.concatenate([jnp.ones(1), 1.0 - alpha + 1e-7]))[:-1]
        weights = alpha * trans
        color = jnp.sum(weights[:, None] * sampled_color, axis=0)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - jnp.sum(weights))

        return color, sampled_color, alpha, weights

    def render_core(
        self,
        ray_o: Float[Array, "3"],
        ray_d: Float[Array, "3"],
        z_vals: Float[Array, "n_sample"],
        background_alpha: Float[Array, "m_sample"] = None,
        background_sampled_color: Float[Array, "m_sample 3"] = None,
        background_rgb: Float[Array, "3"] = None,
        sample_dist: Float = 0.03,
        cos_anneal_ratio: Float = 0.0,
    ) -> tuple[
        Float[Array, "3"],  # color
        Float[Array, "n_sample"],  # sd
        Float[Array, "n_sample 3"],  # sdf_grad
        Float[Array, "n_sample"],  # gradient_err
        Float[Array, "n_sample"],  # s_val
        Float[Array, "m_sample"],  # weights
        Float[Array, "n_sample"],  # cdf
        Float[Array, "n_sample"],  # inside_sphere
        Float[Array, "n_sample"],  # relax_inside_sphere
    ]:
        """Render color for single ray with signed distance.

        Outputs:
            color: [3]
            sd: [n_sample]
            sdf_grad: [n_sample, 3]
            gradient_err: [n_sample]
            s_val: [n_sample]
            weights: [m_sample]
            cdf: [n_sample]
            inside_sphere: [n_sample]
            relax_inside_sphere: [n_sample]
        """
        n_sample = z_vals.shape[0]

        dist = z_vals[1:] - z_vals[:-1]
        dists = jnp.concatenate([dist, jnp.array([sample_dist])])
        mid_z_vals = z_vals + dists * 0.5
        pts = ray_o + ray_d * mid_z_vals[:, None]
        dirs = ray_d[None, :].repeat(n_sample, axis=0)

        h = self.sdf_network(pts)
        sd, feat = h[:, 0], h[:, 1:]
        # work around for grad opertation on batched sdf output
        sdf_func = lambda x: jnp.sum(self.sdf_network(x)[:, 0])
        sdf_grad = jax.vmap(jax.grad(sdf_func))(pts[:, None, :]).reshape(n_sample, 3)

        # get radiance
        sampled_color = self.color_network(pts, sdf_grad, dirs, feat)

        # get variance
        inv_s = self.deviation_network(jnp.zeros(1)).clip(1e-6, 1e6)
        inv_s = inv_s.repeat(n_sample, axis=0)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        true_cos = jnp.sum(dirs * sdf_grad, axis=-1)
        iter_cos = -(
            nn.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + nn.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        estimated_next_sd = sd + iter_cos * dists * 0.5
        estimated_prev_sd = sd - iter_cos * dists * 0.5
        prev_cdf = nn.sigmoid(estimated_prev_sd * inv_s)
        next_cdf = nn.sigmoid(estimated_next_sd * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        pts_norm = jnp.linalg.norm(pts, ord=2, axis=-1)
        inside_sphere = pts_norm < 1.0
        relax_inside_sphere = pts_norm < 1.2

        if background_alpha is not None:
            alpha = (
                background_alpha[:n_sample] * (1 - inside_sphere)
                + alpha * inside_sphere
            )
            alpha = jnp.concatenate([alpha, background_alpha[n_sample:]])
            sampled_color = (
                background_sampled_color[:n_sample] * (1 - inside_sphere[:, None])
                + sampled_color * inside_sphere[:, None]
            )
            sampled_color = jnp.concatenate(
                [sampled_color, background_sampled_color[n_sample:]]
            )

        trans = jnp.cumprod(jnp.concatenate([jnp.ones(1), 1.0 - alpha + 1e-7]))[:-1]
        weights = alpha * trans
        color = jnp.sum(weights[:, None] * sampled_color, axis=0)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - jnp.sum(weights))

        gradient_err = (jnp.linalg.norm(sdf_grad, ord=2, axis=-1) - 1) ** 2

        return (
            color,
            sd,
            sdf_grad,
            gradient_err,
            1.0 / inv_s,
            weights,
            c,
            inside_sphere,
            relax_inside_sphere,
        )

    def __call__(
        self,
        rays_o: Float[Array, "b 3"],
        rays_d: Float[Array, "b 3"],
        near: Float[Array, "b 1"],
        far: Float[Array, "b 1"],
        background_rgb: Float[Array, "3"] = None,
        perturb_overwrite: Float = -1,
        cos_anneal_ratio: Float = 0.0,
    ) -> PyTree:
        """render a batch of rays.

        outputs:
            color_fine: [b, 3]
            s_val: [b, 1]
            cdf: [b, n_samples]
            weight_sum: [b, 1]
            weight_max: [b, 1]
            weights: [b, n_samples]
            sdf_grad: [b, n_samples, 3]
            gradient_err: []
            inside_sphere: [b, n_samples]
        """

        # sample depth values
        batch_size = rays_o.shape[0]
        sample_dist = 2.0 / self.n_samples
        z_vals = jnp.linspace(0, 1, self.n_samples)
        z_vals = near + z_vals * (far - near)

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = jnp.linspace(
                1e-3, 1.0 - 1.0 / (self.n_outside + 1), self.n_outside
            )
        
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = jax.random.uniform(jax.random.PRNGKey(777), (batch_size, 1)) - 0.5
            z_vals = z_vals + t_rand * sample_dist

            if self.n_outside > 0:
                mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = jnp.concatenate([mids, z_vals_outside[..., -1:]], axis=-1)
                lower = jnp.concatenate([z_vals_outside[..., :1], mids], axis=-1)
                t_rand = jax.random.uniform(
                    jax.random.PRNGKey(777), (batch_size, z_vals_outside.shape[-1])
                )
                z_vals_outside = lower + t_rand * (upper - lower)

        if self.n_outside > 0:
            z_vals_outside = (
                far / jnp.flip(z_vals_outside, axis=-1) + 1 / self.n_samples
            )

        # up sampling
        n_samples = self.n_samples
        if self.n_importance > 0:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            sds = self.sdf_network(jax.lax.stop_gradient(pts.reshape(-1, 3)))[:, 0]
            sds = sds.reshape(batch_size, n_samples)

            for i in range(self.up_sample_steps):
                sample_f = partial(
                    self.up_sample,
                    n_importance=self.n_importance // self.up_sample_steps,
                    inv_s=64 * 2**i,
                )
                new_z_vals = jax.vmap(sample_f)(rays_o, rays_d, z_vals, sds)

                z_vals, sds = jax.vmap(
                    partial(self.cat_z_vals, last=i == self.up_sample_steps - 1)
                )(rays_o, rays_d, z_vals, new_z_vals, sds)
            n_samples = self.n_samples + self.n_importance

        # render background
        background_alpha = None
        background_sampled_color = None
        if self.n_outside > 0:
            z_vals_feed = jnp.concatenate([z_vals, z_vals_outside], axis=-1)
            z_vals_feed = jnp.sort(z_vals_feed, axis=-1)
            _, background_sampled_color, background_alpha, _ = jax.vmap(
                partial(self.render_core_outside, sample_dist=sample_dist)
            )(rays_o, rays_d, z_vals_feed)

        # render core
        render_fn = partial(
            self.render_core,
            sample_dist=sample_dist,
            background_rgb=background_rgb,
            cos_anneal_ratio=cos_anneal_ratio,
        )
        (
            color_fine,
            sd,
            sdf_grad,
            gradient_err,
            s_val,
            weights,
            cdf,
            inside_sphere,
            relax_inside_sphere,
        ) = jax.vmap(render_fn)(
            rays_o, rays_d, z_vals, background_alpha, background_sampled_color
        )

        weight_sum = jnp.sum(weights, axis=-1, keepdims=True)
        weight_max = jnp.max(weights, axis=-1, keepdims=True)
        s_val = jnp.mean(s_val, axis=-1, keepdims=True)
        gradient_err = jnp.sum(relax_inside_sphere * gradient_err) / (
            jnp.sum(relax_inside_sphere) + 1e-5
        )

        return {
            "color_fine": color_fine,
            "s_val": s_val,
            "cdf": cdf,
            "weight_sum": weight_sum,
            "weight_max": weight_max,
            "weights": weights,
            "gradients": sdf_grad,
            "gradient_err": gradient_err,
            "inside_sphere": inside_sphere,
        }
