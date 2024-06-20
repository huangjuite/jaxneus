import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory

from utils.dataset import Dataset

from models.fields import SDFNetwork  # RenderingNetwork, SingleVarianceNetwork, NeRF

# from models.renderer import NeuSRenderer

import jax
import jax.numpy as jnp


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
        # self.dataset = Dataset(self.conf["dataset"])
        self.iter_step = 0

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
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network'])
        
        param = self.sdf_network.init(jax.random.PRNGKey(0), jnp.zeros((1, 3)))
        

        # params_to_train = []
        # self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        
        # self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        # self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        # params_to_train += list(self.nerf_outside.parameters())
        # params_to_train += list(self.sdf_network.parameters())
        # params_to_train += list(self.deviation_network.parameters())
        # params_to_train += list(self.color_network.parameters())


if __name__ == "__main__":

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.CRITICAL, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/base.conf")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--mcube_threshold", type=float, default=0.0)
    parser.add_argument("--is_continue", default=False, action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--case", type=str, default="")

    args = parser.parse_args()

    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == "train":
        runner.train()
    elif args.mode == "validate_mesh":
        runner.validate_mesh(
            world_space=True, resolution=512, threshold=args.mcube_threshold
        )
    # TODO
    # elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
    #     _, img_idx_0, img_idx_1 = args.mode.split('_')
    #     img_idx_0 = int(img_idx_0)
    #     img_idx_1 = int(img_idx_1)
    #     runner.interpolate_view(img_idx_0, img_idx_1)
