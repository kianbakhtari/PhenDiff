# Copyright 2023 Thomas Boyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###################################### img2img_comparison.py ######################################
# This script launches a series of experiments to compare class-to-class image transfer methods.
#
# Its config is located in the my_img2img_comparison_conf folder (by default) and managed
# with hydra (https://hydra.cc/).
#
# The experiments are logged with wandb (https://wandb.ai) and run sequentially,
# with metrics computed at the end of each experiment with torch-fidelity.


from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from src.utils_Img2Img import (
    ClassTransferExperimentParams,
    _get_config_path_and_name,
    compute_metrics,
    load_datasets,
    modify_debug_args,
    perform_class_transfer_experiment,
)
from src.utils_misc import setup_logger

logger = get_logger(__name__, log_level="INFO")

torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(
    version_base=None,
    config_path="my_img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # ---------------------------------------- Accelerator ----------------------------------------
    accelerator = Accelerator(
        mixed_precision=cfg.accelerate.launch_args.mixed_precision,
        log_with="wandb",
    )

    # ------------------------------------------- WandB -------------------------------------------
    setup_logger(logger, accelerator)
    logger.info(
        f"Logging to entity/project/run: {cfg.entity}/{cfg.project}/{cfg.run_name}"
    )
    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        # save metadata to the "wandb" directory
        # inside the *parent* folder common to all *experiments*
        init_kwargs={
            "wandb": {
                "entity": cfg.entity,
                "dir": cfg.exp_parent_folder,
                "name": cfg.run_name,
                "save_code": True,
            }
        },
    )

    # ------------------------------------------- Misc. -------------------------------------------
    # get Hydra config & output dir
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    output_dir: str = hydra_cfg["runtime"]["output_dir"]
    # show config
    config_path, config_name = _get_config_path_and_name(cfg, hydra_cfg)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config name: {config_name}")
    logger.info(f"Passed config:\n{OmegaConf.to_yaml(cfg)}")
    # set cache folders
    fidelity_cache_root: Path = Path(cfg.exp_parent_folder, ".fidelity_cache")
    torch_hub_cache_dir = Path(cfg.exp_parent_folder, ".torch_hub_cache")
    torch.hub.set_dir(torch_hub_cache_dir)

    # ------------------------------------------- Debug -------------------------------------------
    num_inference_steps, cfg = modify_debug_args(cfg, logger) # Kian: if not debug, num_inference_steps is None. If debug, num_inference_steps is 10.

    # --------------------------------- Load pretrained pipelines ---------------------------------
    logger.info(f"\033[1m==========================> Loading pipelines\033[0m")
    pipes = call(cfg.pipeline) # Kian: cfg is created from config files. the "call" instanciates the pipeline which is especified by cfg.
    # type(pipes): <class 'omegaconf.dictconfig.DictConfig'>
    # len(pipes): 1

    # manage progress bars
    for pipename in pipes:
        """
        type(pipename): <class 'str'>
        pipename: DDIM
        pipes[pipename]: ConditionalDDIMPipeline {
            "_class_name": "ConditionalDDIMPipeline",
            "_diffusers_version": "0.18.2",
            "scheduler": [
                "diffusers",
                "DDIMScheduler"
            ],
            "unet": [
                "src.cond_unet_2d.cond_unet_2d",
                "CustomCondUNet2DModel"
            ]
        }
        """
        pipes[pipename].set_progress_bar_config(
            position=accelerator.process_index + 1,
            leave=False,
            desc=f"Generating images on process {accelerator.process_index}",
        )

    # ---------------------------------------- Load dataset ---------------------------------------
    # assume only one dataset
    dataset_name = next(iter(cfg.dataset))

    # load dataset TODO: directly instantiate from hydra?
    logger.info(
        f"\033[1m==========================> Loading dataset {dataset_name}\033[0m"
    )
    train_dataset, test_dataset = load_datasets(cfg, dataset_name)
    logger.info(f"Train dataset: {train_dataset}")
    logger.info(f"Test dataset: {test_dataset}")

    # ---------------------------------------- Experiments ----------------------------------------
    # Params common to all experiments
    transfer_exp_common_params = {
        "pipes": pipes,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "cfg": cfg,
        "output_dir": output_dir,
        "accelerator": accelerator,
        "logger": logger,
        "dataset_name": dataset_name,
        "fidelity_cache_root": fidelity_cache_root,
    }

    # Sweep over experiments
    for class_transfer_method in cfg.class_transfer_method: # Kian: cfg.class_transfer_method is ddib
        # args
        exp_args = ClassTransferExperimentParams(
            class_transfer_method=class_transfer_method, # Kian: ddib
            num_inference_steps=num_inference_steps,
            **transfer_exp_common_params,
        )

        ############# Class transfer ############
        logger.info(
            f"\033[1m==========================> Running {class_transfer_method}\033[0m"
        )
        perform_class_transfer_experiment(exp_args)
        accelerator.wait_for_everyone()

        ########## Metrics computation ##########
        logger.info(f"\033[1m==========================> Computing metrics\033[0m")
        if accelerator.is_main_process:
            compute_metrics(exp_args)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
