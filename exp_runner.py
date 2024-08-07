import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
from shutil import copyfile
from tqdm import tqdm, trange
from pyhocon import ConfigFactory
from os.path import join as pjoin
from tensorboardX import SummaryWriter
from functools import partial, cached_property

import jax
import optax
from jax import numpy as jnp
from flax.core import frozen_dict
from jaxtyping import PyTree, Float, Array

KeyArray = Array

from utils.dataset import Dataset
from utils.sdf_pretrain import sdf_pretrain
from utils.visualize import extract_geometry

from models.fields import SDFNetwork, RenderingNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeusRenderer
from models.states import ModelState


class Runner:
    def __init__(
        self,
        conf_path: str,
        mode: str = "train",
        case: str = "CASE_NAME",
        is_continue: bool = False,
    ) -> None:

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace("CASE_NAME", case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf["dataset.data_dir"] = self.conf["dataset.data_dir"].replace(
            "CASE_NAME", case
        )
        self.base_exp_dir = self.conf["general.base_exp_dir"]
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf["dataset"])

        # Training parameters
        self.end_iter = self.conf.get_int("train.end_iter")
        self.save_freq = self.conf.get_int("train.save_freq")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.val_freq = self.conf.get_int("train.val_freq")
        self.val_mesh_freq = self.conf.get_int("train.val_mesh_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        self.validate_resolution_level = self.conf.get_int(
            "train.validate_resolution_level"
        )
        self.learning_rate = self.conf.get_float("train.learning_rate")
        self.learning_rate_alpha = self.conf.get_float("train.learning_rate_alpha")
        self.use_white_bkgd = self.conf.get_bool("train.use_white_bkgd")
        self.warm_up_end = self.conf.get_float("train.warm_up_end", default=0.0)
        self.anneal_end = self.conf.get_float("train.anneal_end", default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float("train.igr_weight")
        self.mask_weight = self.conf.get_float("train.mask_weight")
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.sdf_network = SDFNetwork(**self.conf["model.sdf_network"])
        self.nerf_outside = NeRF(**self.conf["model.nerf"])
        self.color_network = RenderingNetwork(**self.conf["model.rendering_network"])
        self.var_network = SingleVarianceNetwork(**self.conf["model.variance_network"])
        self.renderer = NeusRenderer(
            self.nerf_outside,
            self.sdf_network,
            self.var_network,
            self.color_network,
            **self.conf["model.neus_renderer"]
        )

        # optimizer
        self.optimizer = optax.inject_hyperparams(optax.adam)(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=1e-6,
                peak_value=self.learning_rate,
                warmup_steps=self.warm_up_end,
                decay_steps=self.end_iter,
                end_value=1e-6,
            )
        )

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(pjoin(self.base_exp_dir, "checkpoints"))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == "pkl" and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        self.params_state: ModelState = None
        if latest_model_name is not None:
            logging.critical("Find checkpoint: {}".format(latest_model_name))
            self.params_state = ModelState.load(
                pjoin(self.base_exp_dir, "checkpoints", latest_model_name)
            )
        else:
            self.params_state = self.init()

        # Backup code and Initialize
        if self.mode[:5] == "train":
            self.file_backup()
            self.validate_mesh()
            self.validate_image()

    def init(self, key=jax.random.key(777)) -> ModelState:
        logging.critical("Initializing model")
        # init parameters
        data = self.dataset.gen_random_rays_at(0, 2)
        near, far = self.dataset.near_far_from_sphere(data[0, :3], data[0, 3:6])
        params_ = self.renderer.init(
            key, data[0, :3], data[0, 3:6], near, far, jnp.ones((3))
        )
        opt_state = self.optimizer.init(params_)

        # pretrain sdf
        logging.critical("pretrain sdf sphere")
        param_sdf = sdf_pretrain(self.sdf_network)
        params_["params"] = frozen_dict.copy(
            params_["params"], {"sdf_network": param_sdf["params"]}
        )
        return ModelState(params_, opt_state)

    @cached_property
    def _render(self):
        @jax.jit
        def _inner(
            params, rays_o, rays_d, near, far, background_rgb, cos_anneal_ratio, key
        ):
            rend_fn = partial(
                self.renderer.apply,
                params,
                background_rgb=background_rgb,
                cos_anneal_ratio=cos_anneal_ratio,
                rngs={self.renderer.rng_name: key},
            )
            render_out = jax.vmap(rend_fn)(rays_o, rays_d, near, far)
            return render_out

        return _inner

    @cached_property
    def sdf(self):
        @jax.jit
        def _inner(params: PyTree, x: Float[Array, "b 3"]) -> Float[Array, "b"]:
            return -self.sdf_network.apply(params, x)[:, 0]

        return _inner

    def _loss(
        self, batch_data: Float[Array, "b 10"], key: KeyArray, params: PyTree
    ) -> tuple[Float[Array, ""], PyTree]:
        rays_o, rays_d, true_rgb, mask = (
            batch_data[:, :3],
            batch_data[:, 3:6],
            batch_data[:, 6:9],
            batch_data[:, 9:10],
        )
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = jnp.ones((1, 3))

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).astype(jnp.float32)
        else:
            mask = jnp.ones_like(mask, dtype=jnp.float32)

        mask_sum = mask.sum() + 1e-5
        render_out = self._render(
            params,
            rays_o,
            rays_d,
            near,
            far,
            background_rgb=background_rgb,
            cos_anneal_ratio=self.get_cos_anneal_ratio(),
            key=key,
        )

        color_error = (render_out["color"] - true_rgb) * mask
        color_fine_loss = jnp.abs(color_error).sum() / mask_sum

        grad_norm = jnp.linalg.norm(render_out["gradients"], ord=2, axis=-1)
        grad_err = (grad_norm - 1) ** 2
        grad_maks = render_out["relax_inside_sphere"]
        eikonal_loss = jnp.sum(grad_maks * grad_err) / (jnp.sum(grad_maks) + 1e-5)

        mask_loss = jnp.mean(
            optax.losses.sigmoid_binary_cross_entropy(render_out["weight_sum"], mask)
        )

        loss = (
            color_fine_loss
            + eikonal_loss * self.igr_weight
            + mask_loss * self.mask_weight
        )

        psnr = 20.0 * jnp.log10(
            1.0
            / jnp.sqrt(
                jnp.sum((render_out["color"] - true_rgb) ** 2 * mask) / (mask_sum * 3.0)
            )
        )

        info = {
            "Loss/loss": loss,
            "Loss/color_loss": color_fine_loss,
            "Loss/eikonal_loss": eikonal_loss,
            "Statistics/s_val": jnp.mean(render_out["s_val"]),
            "Statistics/cdf": jnp.sum(render_out["cdf"][:, :1] * mask) / mask_sum,
            "Statistics/weight_max": jnp.sum(render_out["weight_max"] * mask)
            / mask_sum,
            "Statistics/psnr": psnr,
        }
        del render_out

        return loss, info

    def _step(
        self, batch_data: Float[Array, "b 10"], states: ModelState, key: KeyArray
    ) -> PyTree:
        (loss, info), grads = jax.value_and_grad(
            partial(self._loss, batch_data, key), has_aux=True
        )(states.params)

        updates, new_opt_state = self.optimizer.update(grads, states.opt_state)
        new_params = optax.apply_updates(states.params, updates)

        return info, ModelState(new_params, new_opt_state)

    def train(self, key=jax.random.key(777)):
        self.writer = SummaryWriter(log_dir=pjoin(self.base_exp_dir, "logs"))
        iter_step = int(self.params_state.opt_state.count)
        image_perm = self.get_image_perm(key)

        logging.critical("tracing computation graph")
        step_once = jax.jit(self._step)

        for iter_i in trange(0, self.end_iter, initial=iter_step, total=self.end_iter):
            data = self.dataset.gen_random_rays_at(
                image_perm[iter_i % len(image_perm)], self.batch_size
            )
            perturb_rng, key = jax.random.split(key)
            info, self.params_state = step_once(
                jnp.asarray(data), self.params_state, key=perturb_rng
            )
            iter_step = int(self.params_state.opt_state.count)

            # logging
            for k, v in info.items():
                self.writer.add_scalar(k, v, iter_step)

            if iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print(
                    "iter:{:8>d} loss = {} lr={}".format(
                        iter_step,
                        info["Loss/loss"],
                        self.params_state.opt_state.hyperparams["learning_rate"],
                    )
                )

            if iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if iter_step % self.val_freq == 0:
                self.validate_image()

            if iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            if iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm(key)

    def get_image_perm(self, key=jax.random.key(0)):
        return jax.random.permutation(key, self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            iter_step = float(self.params_state.opt_state.count)
            return min(1.0, iter_step / self.anneal_end)

    def validate_mesh(
        self, world_space=False, resolution=64, threshold=0.0, batch_size=2**15
    ):
        param_sdf = {"params": self.params_state.params["params"]["sdf_network"]}
        vertices, triangles = extract_geometry(
            self.dataset.object_bbox_min,
            self.dataset.object_bbox_max,
            resolution,
            threshold=threshold,
            query_func=lambda x: self.sdf(param_sdf, x),
            batch_size=batch_size,
        )
        if world_space:
            vertices = (
                vertices * self.dataset.scale_mats_np[0][0, 0]
                + self.dataset.scale_mats_np[0][:3, 3][None]
            )

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh_dir = pjoin(self.base_exp_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_fp = pjoin(
            mesh_dir, "{:0>8d}.ply".format(self.params_state.opt_state.count)
        )
        mesh.export(mesh_fp)
        logging.critical("Mesh saved at {}".format(mesh_fp))

    def validate_image(self, idx=-1, resolution_level=-1, key=jax.random.key(0)):
        iter_step = int(self.params_state.opt_state.count)
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        print("Validate: iter: {}, camera: {}".format(iter_step, idx))

        rays_o, rays_d = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level
        )
        W, H, _ = rays_o.shape
        rays_o = np.array_split(rays_o.reshape(-1, 3), H)
        rays_d = np.array_split(rays_d.reshape(-1, 3), H)

        out_rgb_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d), total=len(rays_o)):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jnp.ones([3]) if self.use_white_bkgd else None
            perturb_rng, key = jax.random.split(key)
            render_out = self._render(
                self.params_state.params,
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                key=perturb_rng,
            )

            out_rgb_fine.append(render_out["color"])
            n_samples = self.renderer.n_samples + self.renderer.n_importance
            normals = (
                render_out["gradients"] * render_out["weights"][:, :n_samples, None]
            )
            normals = normals * render_out["inside_sphere"][..., None]
            normals = jnp.sum(normals, axis=1)
            out_normal_fine.append(normals)
            del render_out

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0) * 256).clip(0, 255).reshape([H, W, 3])
        )

        normal_img = np.concatenate(out_normal_fine, axis=0)
        rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3])
        normal_img = np.matmul(rot[None, :, :], normal_img[:, :, None])
        normal_img = (128 + 128 * normal_img).clip(0, 255).reshape([H, W, 3])

        os.makedirs(pjoin(self.base_exp_dir, "validations_fine"), exist_ok=True)
        os.makedirs(pjoin(self.base_exp_dir, "normals"), exist_ok=True)

        if len(out_rgb_fine) > 0:
            org_img = self.dataset.image_at(idx, resolution_level=resolution_level)
            cv.imwrite(
                pjoin(
                    self.base_exp_dir,
                    "validations_fine",
                    "{:0>8d}_{}.png".format(iter_step, idx),
                ),
                np.concatenate([img_fine, org_img]),
            )
        if len(out_normal_fine) > 0:
            cv.imwrite(
                pjoin(
                    self.base_exp_dir,
                    "normals",
                    "{:0>8d}_{}.png".format(iter_step, idx),
                ),
                normal_img,
            )

    def render_novel_image(
        self, idx_0, idx_1, ratio, resolution_level, key=jax.random.key(0)
    ):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(
            idx_0, idx_1, ratio, resolution_level=resolution_level
        )
        W, H, _ = rays_o.shape
        rays_o = np.array_split(rays_o.reshape(-1, 3), H)
        rays_d = np.array_split(rays_d.reshape(-1, 3), H)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = jnp.ones([3]) if self.use_white_bkgd else None
            perturb_rng, key = jax.random.split(key)
            render_out = self._render(
                self.params_state.params,
                rays_o_batch,
                rays_d_batch,
                near,
                far,
                background_rgb=background_rgb,
                cos_anneal_ratio=self.get_cos_anneal_ratio(),
                key=perturb_rng,
            )

            out_rgb_fine.append(render_out["color"])

            del render_out

        img_fine = (
            (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256)
            .clip(0, 255)
            .astype(np.uint8)
        )
        return img_fine

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in trange(n_frames):
            images.append(
                self.render_novel_image(
                    img_idx_0,
                    img_idx_1,
                    np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                    resolution_level=4,
                )
            )
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        video_dir = pjoin(self.base_exp_dir, "render")
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        iter_step = int(self.params_state.opt_state.count)
        v_file = pjoin(
            video_dir, "{:0>8d}_{}_{}.mp4".format(iter_step, img_idx_0, img_idx_1)
        )
        writer = cv.VideoWriter(v_file, fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()

    def save_checkpoint(self):
        folder = pjoin(self.base_exp_dir, "checkpoints")
        os.makedirs(folder, exist_ok=True)
        iter_step = self.params_state.opt_state.count
        self.params_state.save(pjoin(folder, "ckpt_{:0>6d}.pkl".format(iter_step)))

    def file_backup(self):
        dir_lis = self.conf["general.recording"]
        os.makedirs(pjoin(self.base_exp_dir, "recording"), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = pjoin(self.base_exp_dir, "recording", dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == ".py":
                    copyfile(pjoin(dir_name, f_name), pjoin(cur_dir, f_name))
        copyfile(self.conf_path, pjoin(self.base_exp_dir, "recording", "config.conf"))


if __name__ == "__main__":

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.CRITICAL, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcube_threshold", type=float, default=0.0)
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--case", type=str, default="")

    args = parser.parse_args()

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate_mesh":
        runner.validate_mesh(
            world_space=True, resolution=512, threshold=args.mcube_threshold
        )
    elif args.mode.startswith(
        "interpolate"
    ):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split("_")
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
