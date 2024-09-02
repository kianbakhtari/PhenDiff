# Copyright 2023 The HuggingFace Team and Thomas Boyer. All rights reserved.
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


import math
import os
from argparse import Namespace
from inspect import signature
from math import ceil
from pathlib import Path
import time
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch_fidelity
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from accelerate.utils import broadcast
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL.Image import Image
from torch import FloatTensor, IntTensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

import wandb

from .cond_unet_2d import CustomCondUNet2DModel
from .custom_embedding import CustomEmbedding
from .custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)
from .pipeline_conditional_ddim import ConditionalDDIMPipeline
from .utils_misc import (
    extract_into_tensor,
    is_it_best_model,
    print_info_at_run_start,
    save_checkpoint,
    split,
)
from .utils_models import SupportedPipelines
from .utils_Img2Img import (
    _ddib_for_training,
    _ddibـfor_testing_during_training
)


def resume_from_checkpoint(
    args: Namespace,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
    num_update_steps_per_epoch: int,
    global_step: int,
    chckpt_save_path: Path,
) -> tuple[int, int, int]:
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
        path = os.path.join(chckpt_save_path, path)
    else:
        # Get the most recent checkpoint
        if not Path.exists(chckpt_save_path) and accelerator.is_main_process:
            logger.warning(
                f"No 'checkpoints' directory found in run folder; creating one."
            )
            os.makedirs(chckpt_save_path)
        accelerator.wait_for_everyone()
        dirs = os.listdir(chckpt_save_path)
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = Path(chckpt_save_path, dirs[-1]).as_posix() if len(dirs) > 0 else None

    if path is None:
        logger.warning(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        first_epoch, resume_step = 0, 0
    else:
        logger.info(f"Resuming from checkpoint {path}")
        print(logger.info(f"\nResuming from checkpoint {path}\n"))
        accelerator.load_state(path)
        global_step = int(path.split("_")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps
        )
    return first_epoch, resume_step, global_step


def get_training_setup(
    args: Namespace,
    accelerator: Accelerator,
    train_dataloader: DataLoader,
    logger: MultiProcessAdapter,
    pipeline_components: list[str],
    components_to_train_transcribed: list[str],
    nb_tot_samples: int,
    nb_tot_samples_raw_ds: int,
    pipeline: SupportedPipelines,
    tot_training_steps: int,
) -> tuple[int, list[int]]:
    """
    Returns
    -------
    - `Tuple[int, List[int]]`
        A tuple containing:
        - the total number of update steps per epoch,
        - the list of actual evaluation batch sizes *for this process*
    """
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    # distribute "batches" for image generation
    # tot_nb_eval_batches is the total number of batches for image generation
    # *across all GPUs* and *per class*
    tot_nb_eval_batches = ceil(args.nb_generated_images / args.eval_batch_size)
    glob_eval_bs = [args.eval_batch_size] * (tot_nb_eval_batches - 1)
    glob_eval_bs += [
        args.nb_generated_images - args.eval_batch_size * (tot_nb_eval_batches - 1)
    ]
    nb_proc = accelerator.num_processes
    actual_eval_batch_sizes_for_this_process = split(
        glob_eval_bs, nb_proc, accelerator.process_index
    )

    print_info_at_run_start(
        logger,
        args,
        pipeline_components,
        components_to_train_transcribed,
        pipeline,
        nb_tot_samples,
        nb_tot_samples_raw_ds,
        total_batch_size,
        tot_training_steps,
    )

    return (num_update_steps_per_epoch, actual_eval_batch_sizes_for_this_process)


def perform_training_epoch(
    num_update_steps_per_epoch: int,
    accelerator: Accelerator,
    pipeline: SupportedPipelines,
    ema_models: dict[str, EMAModel],
    components_to_train_transcribed: list[str],
    epoch: int,
    train_dataloader: DataLoader,
    args: Namespace,
    first_epoch: int,
    resume_step: int,
    global_step: int,
    optimizer,
    lr_scheduler,
    logger: MultiProcessAdapter,
    params_to_clip: list,
    tot_training_steps: int,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    actual_eval_batch_sizes_for_this_process: list[int],
    nb_classes: int,
    dataset,
    raw_dataset,
    full_pipeline_save_folder: Path,
    repo,
    best_metric,
    chckpt_save_path: Path,
    paired_dataloader: DataLoader = None
) -> tuple[int, float | None]:
    # 1. Retrieve models & set then to train mode if applicable
    # components common to all models
    denoiser_model = pipeline.unet
    denoiser_model.train()
    noise_scheduler = pipeline.scheduler

    # components specific to SD
    if args.model_type == "StableDiffusion":
        autoencoder_model = pipeline.vae
        autoencoder_model.train()
        # unwrap the autoencoder before calling it
        autoencoder_model = accelerator.unwrap_model(autoencoder_model)
        class_embedding = pipeline.class_embedding
        class_embedding.train()
    else:
        autoencoder_model = None
        class_embedding = None

    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in enumerate(train_dataloader):
        # stop at tot_training_steps
        if global_step >= tot_training_steps:
            break

        if args.debug:
            # stop at 10 steps for quick debug purposes
            # TODO: this might break resume_from_checkpoint just below?
            if step >= 10:
                logger.warning("Debug flag: stopping after 10 steps")
                break

        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update()
            continue

        if args.use_pytorch_loader:
            clean_images = batch[0]
            class_labels = batch[1]
        else:
            clean_images = batch["images"]
            class_labels = batch["class_labels"]

        if args.model_type == "StableDiffusion":
            # Convert images to latent space
            clean_images = autoencoder_model.encode(clean_images).latent_dist.sample()  # type: ignore
            clean_images = clean_images * autoencoder_model.config.scaling_factor  # type: ignore
            # TODO: ↑ why?

        # Sample noise that we'll add to the images
        noise: FloatTensor = torch.randn(clean_images.shape).to(clean_images.device)  # type: ignore

        # Sample a random timestep for each image
        timesteps: IntTensor = torch.randint(  # type: ignore
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Unconditional pass for CLF guidance training
        # synchronize the conditional & unconditional passes of CLF guidance training
        # between all the processes to circumvent a nasty and unexplained bug...
        # fill tensor on main proc
        if accelerator.is_main_process:
            # always true if proba_uncond == 1 as torch.rand -> [0;1]
            do_uncond_pass_across_all_procs = (
                torch.tensor(1, device=accelerator.device)
                if torch.rand(1) < args.proba_uncond
                else torch.tensor(0, device=accelerator.device)
            )
        else:
            do_uncond_pass_across_all_procs = torch.tensor(0, device=accelerator.device)
            
        accelerator.wait_for_everyone()
        # broadcast tensor to all procs
        do_uncond_pass_across_all_procs = broadcast(do_uncond_pass_across_all_procs)

        do_unconditional_pass: bool = do_uncond_pass_across_all_procs.item()  # type: ignore
        accelerator.log(
            {
                "unconditional step": int(do_unconditional_pass),
            },
            step=global_step,
        )

        # with accelerator.accumulate(denoiser_model): # TODO: fix grad accumulation for multiple models
        loss_value = _diffusion_and_backward(
            args=args,
            accelerator=accelerator,
            global_step=global_step,
            denoiser_model=denoiser_model,
            noisy_images=noisy_images,
            timesteps=timesteps,
            class_labels=class_labels,
            noise=noise,
            noise_scheduler=noise_scheduler,
            clean_images=clean_images,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            class_embedding=class_embedding,
            do_unconditional_pass=do_unconditional_pass,
            logger=logger,
            params_to_clip=params_to_clip,
        )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step = _syn_training_state(
                args=args,
                pipeline=pipeline,
                components_to_train_transcribed=components_to_train_transcribed,
                ema_models=ema_models,
                progress_bar=progress_bar,
                global_step=global_step,
                accelerator=accelerator,
                logger=logger,
                chckpt_save_path=chckpt_save_path,
            )

        # log some values & define W&B alert
        logs = {
            "loss": loss_value,
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }
        if args.use_ema:
            logs["ema_decay"] = list(ema_models.values())[0].cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        if math.isnan(loss_value) and accelerator.is_main_process:
            msg = f"Loss is NaN at step {global_step} / epoch {epoch}"
            wandb.alert(
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)

        # Generate sample images for visual inspection & metrics computation
        if (
            args.eval_save_model_every_opti_steps is not None
            and global_step % args.eval_save_model_every_opti_steps == 0
        ):
            best_metric = generate_samples_compute_metrics_save_pipe(
                args,
                accelerator,
                pipeline,
                image_generation_tmp_save_folder,
                fidelity_cache_root,
                actual_eval_batch_sizes_for_this_process,
                epoch,
                global_step,
                ema_models,
                components_to_train_transcribed,
                nb_classes,
                logger,
                dataset,
                raw_dataset,
                best_metric,
                full_pipeline_save_folder,
                repo,
            )

    progress_bar.close()

    # wait for everybody at end of each training epoch
    accelerator.wait_for_everyone()

    return global_step, best_metric


def paired_dataset_visual_inspection_and_log(
        accelerator: Accelerator,
        global_step: int,
        source_images: torch.tensor,
        target_images: torch.tensor,
        translated_images: torch.tensor,
        split: str
):
    def process(tensor: torch.tensor) -> np.ndarray:
            return tensor.detach()[:10].permute(0, 2, 3, 1).cpu().numpy()

    if accelerator.is_main_process:
        source_images_processed_for_logging = process(source_images)
        target_images_processed_for_logging = process(target_images)
        translated_images_processed_for_logging = process(translated_images)

        source_images_to_log = [
            wandb.Image(
                source_images_processed_for_logging[i],
                caption=f"{split} - source image {i}",
            )
            for i in range(len(source_images_processed_for_logging))
        ]

        target_images_to_log = [
            wandb.Image(
                target_images_processed_for_logging[i],
                caption=f"{split} - target image {i}",
            )
            for i in range(len(target_images_processed_for_logging))
        ]

        translated_images_to_log = [
            wandb.Image(
                translated_images_processed_for_logging[i],
                caption=f"{split} - translated image {i}",
            )
            for i in range(len(translated_images_processed_for_logging))
        ]
        
        data_to_log = [
            [src, tar, trans] for src, tar, trans in zip(
                source_images_to_log,
                target_images_to_log,
                translated_images_to_log
            )
        ]
        table = wandb.Table(
            ["source images", "target images", "translated images"],
            data_to_log,
        )

        accelerator.log(
            {
                f"{split} data samples - gs: {global_step}": table
            },
            step=global_step
        )


def perform_class_transfer_for_paired_training(
    num_update_steps_per_epoch: int,
    accelerator: Accelerator,
    pipeline: ConditionalDDIMPipeline,
    ema_models: dict[str, EMAModel],
    components_to_train_transcribed: list[str],
    epoch: int,
    dataloader: DataLoader,
    args: Namespace,
    first_epoch: int,
    resume_step: int,
    global_step: int,
    optimizer,
    lr_scheduler,
    logger: MultiProcessAdapter,
    params_to_clip: list,
    tot_training_steps: int,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    actual_eval_batch_sizes_for_this_process: list[int],
    nb_classes: int,
    full_pipeline_save_folder: Path,
    repo,
    best_metric,
    chckpt_save_path: Path,
) -> tuple[int, float | None]:
    
    def make_tensor_binary(tensor: torch.Tensor) -> torch.Tensor:
        # The input tensor should be centered around 0
        tensor = torch.sigmoid(tensor)
        tensor = torch.round(tensor)
        assert torch.all((tensor == 0) | (tensor == 1)), "tensor should be binary."
        return tensor
        
    pipe = pipeline
    denoiser_model = pipeline.unet
    denoiser_model.train()
    num_inference_steps = 30
    # ----------- Distributed inference & device placement ----------- #
    dataloader = accelerator.prepare(dataloader)
    # pipe = pipe.to(accelerator.device)

    # ----------- Creat save directories ----------- #
    accelerator.wait_for_everyone()

    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")
    
    # ----------- Iterate Over Batches ----------- #
    do_visual_inspection_and_log = False

    sources_accu = []
    targets_accu = []
    translateds_accu = []

    for step, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"Running paired translation for training, iterating over batches...",
            position=0,
        )
    ):  
        source_images = batch["source_images"].to(accelerator.device)
        source_labels = batch["source_class"].to(accelerator.device)
        target_images = batch["target_images"].to(accelerator.device)
        target_labels = batch["target_class"].to(accelerator.device)

        orig_class_labels = source_labels
        target_class_labels = 1 - orig_class_labels
        assert torch.equal(target_class_labels, target_labels), "target_class_labels and target_labels should be the same."

        translated_images = _ddib_for_training(
            pipe,
            source_images, # [bs, ch, 128, 128]
            orig_class_labels, # [bs] --> [1, 1, 1, 1, 1, 1, ...]
            target_class_labels,
            num_inference_steps,
            process_idx=accelerator.process_index
        )

        if torch.isnan(translated_images).any():
            msg = "NaN values found in translated_images!"
            logger.warn(f"\n{msg}")
            accelerator.log({'warning': msg})
        if torch.isinf(translated_images).any():
            msg = "Inf values found in translated_images!"
            logger.warn(f"\n{msg}")
            accelerator.log({'warning': msg})

        clampped_translated_images = torch.clamp(translated_images, -1, 1)
        mse_loss = F.mse_loss(clampped_translated_images, target_images)
        bce_loss_value, dice = None, None # Default

        if args.paired_training_loss.lower().strip() == "bce":
            assert all(value in [-1, 1] for value in target_images.unique()), "target_images should be binary in {-1, 1}"
            target_images = make_tensor_binary(target_images) # Going from {-1, 1} to {0, 1}
            num_positive = (target_images == 1).sum().item()
            num_negative = (target_images == 0).sum().item()
            pos_weight = torch.tensor([num_negative / (num_positive + 1e-3)], dtype=torch.float32).to(accelerator.device)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(translated_images, target_images)
            accelerator.backward(bce_loss)
            loss_value, bce_loss_value = bce_loss, bce_loss.item()

            translated_images = make_tensor_binary(translated_images)
            dice = dice_score(pred=translated_images, target=target_images)
        else:
            accelerator.backward(mse_loss)
            loss_value = mse_loss

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if len(sources_accu) < 20:
            sources_accu.append(source_images)
            targets_accu.append(target_images)
            translateds_accu.append(translated_images)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step = _syn_training_state(
                args=args,
                pipeline=pipeline,
                components_to_train_transcribed=components_to_train_transcribed,
                ema_models=ema_models,
                progress_bar=progress_bar,
                global_step=global_step,
                accelerator=accelerator,
                logger=logger,
                chckpt_save_path=chckpt_save_path,
            )

        logs = {
            "train_bce": bce_loss_value,
            "train_mse": mse_loss,
            "train_dice": dice,
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }

        if args.use_ema:
            logs["ema_decay"] = list(ema_models.values())[0].cur_decay_value

        # progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        if math.isnan(loss_value) and accelerator.is_main_process:
            msg = f"Loss is NaN at step {global_step} / epoch {epoch}"
            wandb.alert(
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)

        if global_step % args.visual_inspection_interval == 0:
            do_visual_inspection_and_log = True

    progress_bar.close()

    if do_visual_inspection_and_log:
        print("Visual evaluation on train set...")
        paired_dataset_visual_inspection_and_log(
            accelerator=accelerator,
            global_step=global_step,
            source_images=torch.cat(sources_accu, dim=0),
            target_images=torch.cat(targets_accu, dim=0),
            translated_images=torch.cat(translateds_accu, dim=0),
            split="train"
        )
    
    accelerator.wait_for_everyone()
    return global_step, best_metric, loss_value


def perform_sample_prediction_for_paired_training(
    num_update_steps_per_epoch: int,
    accelerator: Accelerator,
    pipeline: SupportedPipelines,
    ema_models: dict[str, EMAModel],
    components_to_train_transcribed: list[str],
    epoch: int,
    dataloader: DataLoader,
    args: Namespace,
    first_epoch: int,
    resume_step: int,
    global_step: int,
    optimizer,
    lr_scheduler,
    logger: MultiProcessAdapter,
    params_to_clip: list,
    tot_training_steps: int,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    actual_eval_batch_sizes_for_this_process: list[int],
    nb_classes: int,
    dataset,
    raw_dataset,
    full_pipeline_save_folder: Path,
    repo,
    best_metric,
    chckpt_save_path: Path,
) -> tuple[int, float | None]:
    
    denoiser_model = pipeline.unet
    denoiser_model.train()
    noise_scheduler = pipeline.scheduler
    
    autoencoder_model = None
    class_embedding = None

    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch} - In Sample Prediction For Paired Training")

    for step, batch in enumerate(dataloader):
        if global_step >= tot_training_steps:
            break

        # Skip steps until we reach the resumed step
        # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
        #     if step % args.gradient_accumulation_steps == 0:
        #         progress_bar.update()
        #     continue

        source_images = batch["source_images"].to(accelerator.device)
        source_classes = batch["source_class"].to(accelerator.device)
        target_images = batch["target_images"].to(accelerator.device)
        target_classes = batch["target_class"].to(accelerator.device)

        noise: FloatTensor = torch.randn(source_images.shape).to(source_images.device)  # type: ignore

        
        timesteps: IntTensor = torch.randint(  # type: ignore
            0,
            noise_scheduler.config.num_train_timesteps,
            (source_images.shape[0],),
            device=source_images.device,
        ).long()
        
        noisy_source_images = noise_scheduler.add_noise(source_images, noise, timesteps)

        loss_value = _diffusion_and_backward_for_paired_sample_prediction(
            args=args,
            accelerator=accelerator,
            global_step=global_step,
            denoiser_model=denoiser_model,
            timesteps=timesteps,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            class_embedding=class_embedding,
            logger=logger,
            params_to_clip=params_to_clip,
            noise_scheduler=noise_scheduler,
            noise=noise,
            noisy_source_images=noisy_source_images,
            clean_source_images=source_images,
            target_images=target_images,
            source_classes=source_classes,
            target_classes=target_classes,
        )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step = _syn_training_state(
                args=args,
                pipeline=pipeline,
                components_to_train_transcribed=components_to_train_transcribed,
                ema_models=ema_models,
                progress_bar=progress_bar,
                global_step=global_step,
                accelerator=accelerator,
                logger=logger,
                chckpt_save_path=chckpt_save_path,
            )

        # log some values & define W&B alert
        logs = {
            "loss": loss_value,
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }
        if args.use_ema:
            logs["ema_decay"] = list(ema_models.values())[0].cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        if math.isnan(loss_value) and accelerator.is_main_process:
            msg = f"Loss is NaN at step {global_step} / epoch {epoch}"
            wandb.alert(
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)

        # Generate sample images for visual inspection & metrics computation
        if (
            args.eval_save_model_every_opti_steps is not None
            and global_step % args.eval_save_model_every_opti_steps == 0
        ):
            best_metric = generate_samples_compute_metrics_save_pipe(
                args,
                accelerator,
                pipeline,
                image_generation_tmp_save_folder,
                fidelity_cache_root,
                actual_eval_batch_sizes_for_this_process,
                epoch,
                global_step,
                ema_models,
                components_to_train_transcribed,
                nb_classes,
                logger,
                dataset,
                raw_dataset,
                best_metric,
                full_pipeline_save_folder,
                repo,
            )

    progress_bar.close()

    # wait for everybody at end of each training epoch
    accelerator.wait_for_everyone()

    return global_step, best_metric, loss_value


def _diffusion_and_backward(
    args: Namespace,
    accelerator: Accelerator,
    global_step: int,
    denoiser_model: UNet2DConditionModel | CustomCondUNet2DModel,
    noisy_images: torch.Tensor,
    timesteps: IntTensor,
    class_labels: torch.Tensor,
    noise: FloatTensor,
    noise_scheduler: DDIMScheduler,
    clean_images: FloatTensor,
    optimizer,
    lr_scheduler,
    class_embedding: CustomEmbedding | None,
    do_unconditional_pass: bool,
    logger: MultiProcessAdapter,
    params_to_clip: list,
) -> float:
    # 1. Obtain model prediction
    match args.model_type:
        case "StableDiffusion":
            model_output = _SD_prediction_wrapper(
                accelerator=accelerator,
                do_unconditional_pass=do_unconditional_pass,
                class_labels=class_labels,
                args=args,
                class_embedding=class_embedding,  # type: ignore
                denoiser_model=denoiser_model,  # type: ignore
                noisy_images=noisy_images,
                timesteps=timesteps,
            )
        case "DDIM":
            model_output = _DDIM_prediction_wrapper(
                accelerator,
                do_unconditional_pass,
                class_labels,
                denoiser_model,  # type: ignore
                noisy_images,
                timesteps,
            )
        case _:
            raise ValueError(f"Unsupported model type: {args.model_type}")

    # 2. Compute loss
    match noise_scheduler.config.prediction_type:  # type: ignore
        case "epsilon":
            loss = F.mse_loss(model_output, noise)
        case "sample":
            alpha_t = extract_into_tensor(
                noise_scheduler.alphas_cumprod,
                timesteps,
                (clean_images.shape[0], 1, 1, 1),
            )
            snr_weights = alpha_t / (1 - alpha_t)
            loss = snr_weights * F.mse_loss(
                model_output, clean_images, reduction="none"
            )  # use SNR weighting from distillation paper
            loss = loss.mean()
        case "v_prediction":
            # Kian: our case.
            velocity = noise_scheduler.get_velocity(clean_images, noise, timesteps)
            loss = F.mse_loss(model_output, velocity)
        case _:
            raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

    # 3. Compute & clip gradients
    accelerator.backward(loss)

    if accelerator.sync_gradients:
        gard_norm = accelerator.clip_grad_norm_(params_to_clip, 1.0)
        accelerator.log({"gradient norm": gard_norm}, step=global_step)
        if gard_norm.isnan().any() and accelerator.is_main_process:  # type: ignore
            msg = f"Gradient norm is NaN at step {global_step}"
            wandb.alert(
                title="NaN gradient norm",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)

    # 4. Perform optimization step
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def _diffusion_and_backward_for_paired_sample_prediction(
    args: Namespace,
    accelerator: Accelerator,
    global_step: int,
    denoiser_model: UNet2DConditionModel | CustomCondUNet2DModel,
    timesteps: IntTensor,
    lr_scheduler,
    optimizer,
    class_embedding: CustomEmbedding | None,
    logger: MultiProcessAdapter,
    params_to_clip: list,
    noise_scheduler: DDIMScheduler,
    noise: FloatTensor,
    noisy_source_images: FloatTensor,
    clean_source_images: FloatTensor,
    target_images: FloatTensor,
    source_classes: torch.Tensor,
    target_classes: torch.Tensor
) -> float:
    """
    This is a modified version of _diffusion_and_backward
    function specifically for performing diffusion and backward
    on the paired dataset, if provided.
    """

    def predict_x0_from_noisy_sample(
        ddim_sheduler: DDIMScheduler,
        noisy_samples: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        This function is intended to be the reverse of add_noise function of our DDIMscheduler
        see: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py#L471
        """

        alphas_cumprod = ddim_sheduler.alphas_cumprod.to(device=noisy_samples.device)
        # alphas_cumprod = ddim_sheduler.alphas_cumprod.to(dtype=noisy_samples.dtype)
        timesteps = timesteps.to(noisy_samples.device)
        predicted_noise = predicted_noise.to(noisy_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(noisy_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        estimation_of_x0 = (noisy_samples - (sqrt_one_minus_alpha_prod * predicted_noise)) / sqrt_alpha_prod
        return estimation_of_x0


    match args.model_type:
        case "StableDiffusion":
            raise NotImplementedError()
        case "DDIM":
            model_output = _DDIM_prediction_wrapper(
                accelerator=accelerator,
                do_unconditional_pass=False,
                class_labels=target_classes,
                denoiser_model=denoiser_model,
                noisy_images=noisy_source_images,
                timesteps=timesteps,
            )
        case _:
            raise ValueError(f"Unsupported model type: {args.model_type}")
    
    predicted_x0 = predict_x0_from_noisy_sample(
        ddim_sheduler=noise_scheduler,
        noisy_samples=noisy_source_images,
        predicted_noise=model_output,
        timesteps=timesteps
    )

    loss = F.mse_loss(predicted_x0, target_images)
    loss = loss.mean()
    accelerator.backward(loss)

    if accelerator.sync_gradients:
        gard_norm = accelerator.clip_grad_norm_(params_to_clip, 1.0)
        accelerator.log({"gradient norm": gard_norm}, step=global_step)
        if gard_norm.isnan().any() and accelerator.is_main_process:  # type: ignore
            msg = f"Gradient norm is NaN at step {global_step}"
            wandb.alert(
                title="NaN gradient norm",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def _SD_prediction_wrapper(
    accelerator: Accelerator,
    do_unconditional_pass: bool,
    class_labels: torch.Tensor,
    args: Namespace,
    class_embedding: CustomEmbedding,
    denoiser_model: UNet2DConditionModel,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    # classifier-free guidance: randomly discard conditioning to train unconditionally
    if do_unconditional_pass:
        class_emb = torch.zeros(
            (
                class_labels.shape[0],
                77,  # TODO: hardcoded dumb repeat, fix this
                args.class_embedding_dim,
            )
        ).to(accelerator.device)
    else:
        class_emb = class_embedding(class_labels)
        # TODO: hardcoded dumb cast, fix this
        (bs, ed) = class_emb.shape
        class_emb = class_emb.reshape(bs, 1, ed)
        padding = torch.zeros_like(class_emb).repeat(1, 76, 1).to(class_emb.device)
        class_emb = torch.cat([class_emb, padding], dim=1)

    # use the class embedding as the "encoder hidden state"
    encoder_hidden_states = class_emb

    # obtain the model prediction
    model_output = denoiser_model(
        sample=noisy_images,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    return model_output


def _DDIM_prediction_wrapper(
    accelerator: Accelerator,
    do_unconditional_pass: bool,
    class_labels: torch.Tensor,
    denoiser_model: CustomCondUNet2DModel,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
):
    # classifier-free guidance: randomly discard conditioning to train unconditionally
    if do_unconditional_pass:
        class_emb = torch.zeros(
            (
                class_labels.shape[0],
                accelerator.unwrap_model(denoiser_model).time_embed_dim,
            )
        ).to(accelerator.device)
        class_labels = None  # type: ignore
    else:
        class_emb = None

    # Predict the noise residual
    sig = signature(accelerator.unwrap_model(denoiser_model).forward)
    if "class_emb" in sig.parameters:
        model_output = denoiser_model(
            sample=noisy_images,
            timestep=timesteps,
            class_labels=class_labels,
            class_emb=class_emb,
        ).sample
    else:
        assert (
            not do_unconditional_pass
        ), "'do_unconditional_pass' is True but the denoiser model does not take a 'class_emb' argument"
        model_output = denoiser_model(
            sample=noisy_images,
            timestep=timesteps,
            class_labels=class_labels,
        ).sample

    return model_output


def _syn_training_state(
    args: Namespace,
    pipeline: SupportedPipelines,
    components_to_train_transcribed: list[str],
    ema_models: dict[str, EMAModel],
    progress_bar,
    global_step: int,
    accelerator: Accelerator,
    logger: MultiProcessAdapter,
    chckpt_save_path: Path,
) -> int:
    # update the EMA weights if applicable
    if args.use_ema:
        for module_name, module in pipeline.components.items():
            if module_name in components_to_train_transcribed:
                ema_models[module_name].step(module.parameters())

    # update the step counters
    progress_bar.update()
    global_step += 1

    if global_step % args.checkpointing_steps == 0:
        # time to save a checkpoint!
        save_checkpoint(
            chckpt_save_path=chckpt_save_path,
            global_step=global_step,
            accelerator=accelerator,
            logger=logger,
            args=args,
        )

    return global_step


@torch.no_grad()
def generate_samples_compute_metrics_save_pipe(
    args: Namespace,
    accelerator: Accelerator,
    pipeline: SupportedPipelines,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    actual_eval_batch_sizes_for_this_process: list[int],
    epoch: int,
    global_step: int,
    ema_models: dict[str, EMAModel],
    components_to_train_transcribed: list[str],
    nb_classes: int,
    logger: MultiProcessAdapter,
    dataset: ImageFolder | Subset,
    raw_dataset: ImageFolder | Subset,
    best_metric: float | None,
    full_pipeline_save_folder: Path,
    repo,
) -> float | None:
    # try to clear cache before gen
    torch.cuda.empty_cache()
    # generate samples and compute metrics
    best_model_to_date, best_metric = _generate_samples_and_compute_metrics(
        args=args,
        accelerator=accelerator,
        pipeline=pipeline,
        image_generation_tmp_save_folder=image_generation_tmp_save_folder,
        fidelity_cache_root=fidelity_cache_root,
        actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
        epoch=epoch,
        global_step=global_step,
        ema_models=ema_models,
        components_to_train_transcribed=components_to_train_transcribed,
        nb_classes=nb_classes,
        logger=logger,
        dataset=dataset,
        raw_dataset=raw_dataset,
        best_metric=best_metric if accelerator.is_main_process else None,
    )
    # save model if best to date
    if accelerator.is_main_process:
        # log best model indicator
        accelerator.log(
            {
                "best_model_to_date": int(best_model_to_date),  # type: ignore
            },
            step=global_step,
        )
        # save pipeline if best model to date
        if epoch != 0 and best_model_to_date:
            save_pipeline(
                accelerator=accelerator,
                args=args,
                pipeline=pipeline,
                full_pipeline_save_folder=full_pipeline_save_folder,
                repo=repo,
                epoch=epoch,
                logger=logger,
                ema_models=ema_models,
                components_to_train_transcribed=components_to_train_transcribed,
            )
        # return new best metric value
        return best_metric
    else:
        return None


@torch.no_grad()
def _generate_samples_and_compute_metrics(
    args: Namespace,
    accelerator: Accelerator,
    pipeline: SupportedPipelines,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    actual_eval_batch_sizes_for_this_process: list[int],
    epoch: int,
    global_step: int,
    ema_models: dict[str, EMAModel],
    components_to_train_transcribed: list[str],
    nb_classes: int,
    logger: MultiProcessAdapter,
    dataset: ImageFolder | Subset,
    raw_dataset: ImageFolder | Subset,
    best_metric: float | None,
) -> tuple[bool | None, float | None]:
    # 1. Progress bar
    progress_bar = tqdm(
        total=len(actual_eval_batch_sizes_for_this_process),
        desc=f"Generating images on process {accelerator.process_index}",
    )

    # 2. Use the EMA weights if applicable
    pipeline_components = {}
    for module_name, module in pipeline.components.items():
        if module_name in components_to_train_transcribed and args.use_ema:
            # this module is EMA'ed; extract it
            unwrapped_module = accelerator.unwrap_model(module)
            # store its parameters in the EMA model
            ema_models[module_name].store(unwrapped_module.parameters())
            # temporarily copy the EMA parameters to the module
            ema_models[module_name].copy_to(unwrapped_module.parameters())
            # and save that module to the inference pipeline
            pipeline_components[module_name] = accelerator.unwrap_model(
                unwrapped_module
            )
        else:
            pipeline_components[module_name] = accelerator.unwrap_model(module)

    # 3. Create inference pipeline
    match args.model_type:
        case "StableDiffusion":
            inference_pipeline = CustomStableDiffusionImg2ImgPipeline(
                **pipeline_components
            )
        case "DDIM":
            inference_pipeline = ConditionalDDIMPipeline(**pipeline_components)
        case _:
            raise ValueError(f"Unknown model type {args.model_type}")
    inference_pipeline.set_progress_bar_config(disable=True)

    # 4. Miscellaneous preparations
    # set manual seed in order to observe the outputs produced from the same Gaussian noise
    generator = torch.Generator(device=accelerator.device).manual_seed(5742877512)

    # list containing the values of the main metric (defined by user) for each class
    # (only instantiated on main process as metrics are themselves computed on main process only)
    if accelerator.is_main_process:
        main_metric_values = []

    # only one gen pass if unconditional model
    if args.proba_uncond == 1:
        nb_classes = 1

    # 5. Run pipeline in inference (sample random noise and denoise)
    # Generate args.nb_generated_images in batches *per class*
    # for metrics computation
    for class_label in range(nb_classes):
        # get class name
        if args.proba_uncond == 1:
            class_name = "unconditional"
        else:
            class_name = dataset.classes[class_label]  # type: ignore

        # clean image_generation_tmp_save_folder (it's per-class)
        if accelerator.is_local_main_process:
            if os.path.exists(image_generation_tmp_save_folder):
                rmtree(image_generation_tmp_save_folder)
            os.makedirs(image_generation_tmp_save_folder, exist_ok=False)
        accelerator.wait_for_everyone()

        # pretty bar
        postfix_str = f"Current class: {class_name} ({class_label+1}/{nb_classes})"
        progress_bar.set_postfix_str(postfix_str)

        # generate and save to disk all images for this class
        match args.model_type:
            case "StableDiffusion":
                _generate_save_images_for_this_class_SD(
                    accelerator=accelerator,
                    image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                    class_label=class_label,
                    class_name=class_name,
                    progress_bar=progress_bar,
                    actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                    args=args,
                    generator=generator,
                    pipeline=inference_pipeline,  # type: ignore
                    epoch=epoch,
                    global_step=global_step,
                )
            case "DDIM":
                _generate_save_images_for_this_class_DDIM(
                    accelerator=accelerator,
                    image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                    class_label=class_label,
                    class_name=class_name,
                    progress_bar=progress_bar,
                    actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                    args=args,
                    generator=generator,
                    pipeline=inference_pipeline,  # type: ignore
                    epoch=epoch,
                    global_step=global_step,
                )
            case _:
                raise ValueError(f"Invalid model type: {args.model_type}")
        # wait for all processes to finish generating+saving images
        # before computing metrics
        accelerator.wait_for_everyone()
        # try to clear cache after gen
        torch.cuda.empty_cache()

        # compute metrics on main process for this class
        if accelerator.is_main_process:
            _compute_log_metrics(
                logger=logger,
                class_name=class_name,
                class_label=class_label,
                args=args,
                image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                fidelity_cache_root=fidelity_cache_root,
                accelerator=accelerator,
                global_step=global_step,
                main_metric_values=main_metric_values,  # type: ignore
                raw_dataset=raw_dataset,
            )

        # resync everybody for each class
        accelerator.wait_for_everyone()

    # 6. Check if it is the best model to date
    if accelerator.is_main_process:
        best_model_to_date, best_metric = is_it_best_model(
            main_metric_values, best_metric, logger, args  # type: ignore
        )
    else:
        best_model_to_date = None
        best_metric = None

    return best_model_to_date, best_metric


@torch.no_grad()
def _generate_save_images_for_this_class_SD(
    accelerator: Accelerator,
    image_generation_tmp_save_folder: Path,
    class_label: int,
    class_name: str,
    progress_bar,
    actual_eval_batch_sizes_for_this_process: list[int],
    args: Namespace,
    generator: torch.Generator,
    pipeline: CustomStableDiffusionImg2ImgPipeline,
    epoch: int,
    global_step: int,
) -> None:
    # loop over eval batches for this process
    for batch_idx, actual_bs in enumerate(actual_eval_batch_sizes_for_this_process):
        images, latents = pipeline(  # pyright: ignore reportGeneralTypeIssues
            image=None,  # start generation from pure Gaussian noise
            latent_shape=(actual_bs, 4, 16, 16),  # TODO: parametrize
            class_labels=torch.tensor(
                [class_label] * actual_bs, device=accelerator.device
            ).long(),
            strength=1,  # no noise will actually be added if image=None
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_factor,
            generator=generator,
            output_type="np+latent",
            device=accelerator.device,
        )

        # save images to disk
        images_pil: list[Image] = pipeline.numpy_to_pil(images)
        for idx, img in enumerate(images_pil):
            tot_idx = args.eval_batch_size * batch_idx + idx
            filename = f"process_{accelerator.local_process_index}_sample_{tot_idx}.png"
            assert not Path(
                filename
            ).exists(), "Rewriting existing generated image file!"
            img.save(
                Path(
                    image_generation_tmp_save_folder,
                    filename,
                )
            )

        # denormalize the images/latents and save to logger if first batch
        # (first batch of main process only to prevent "logger overflow")
        # (and actualy limit to 50 images/latents maximum)
        if batch_idx == 0 and accelerator.is_main_process:
            # log images
            images_processed = (images * 255).round().astype("uint8")  # type: ignore
            accelerator.log(
                {
                    f"generated_samples/{class_name}": [
                        wandb.Image(img) for img in images_processed[:50]
                    ],
                    "epoch": epoch,
                },
                step=global_step,
            )
            # log latents
            assert latents.shape == (actual_bs, 4, 16, 16)  # hardcoded above
            latents_processed = latents.mean(
                dim=1, keepdim=True
            )  # mean over the channels
            latents_processed -= latents_processed.amin(
                dim=(2, 3), keepdim=True
            )  # min-max norm
            latents_processed /= latents_processed.amax(dim=(2, 3), keepdim=True)
            latents_processed = latents_processed.cpu().numpy()
            latents_processed = (latents_processed * 255).round().astype("uint8")
            accelerator.log(
                {
                    f"generated_latents/{class_name}": [
                        wandb.Image(lat) for lat in latents_processed[:50]
                    ],
                },
                step=global_step,
            )
        progress_bar.update()


def _generate_save_images_for_this_class_DDIM(
    accelerator: Accelerator,
    image_generation_tmp_save_folder: Path,
    class_label: int,
    class_name: str,
    progress_bar,
    actual_eval_batch_sizes_for_this_process: list[int],
    args: Namespace,
    generator: torch.Generator,
    pipeline: ConditionalDDIMPipeline,
    epoch: int,
    global_step: int,
):
    # loop over eval batches for this process
    for batch_idx, actual_bs in enumerate(actual_eval_batch_sizes_for_this_process):
        if args.proba_uncond == 1:
            class_labels = None
            class_emb = torch.zeros(
                (actual_bs, accelerator.unwrap_model(pipeline.unet).time_embed_dim)
            ).to(accelerator.device)
        else:
            class_labels = torch.full(
                (actual_bs,), class_label, device=pipeline.device
            ).long()
            class_emb = None
        
        images = pipeline(
            for_testing=True, # Kian: if we are here during unpaired training, this value doesn't have any effect since _original_no_grad_call will be called
            class_labels=class_labels,
            class_emb=class_emb,
            w=args.guidance_factor,
            generator=generator,
            num_inference_steps=args.num_inference_steps,
            output_type="numpy",
        ).images  # type: ignore

        # save images to disk
        images_pil: list[Image] = pipeline.numpy_to_pil(images)
        for idx, img in enumerate(images_pil):
            tot_idx = args.eval_batch_size * batch_idx + idx
            filename = f"process_{accelerator.local_process_index}_sample_{tot_idx}.png"
            assert not Path(
                filename
            ).exists(), "Rewriting existing generated image file!"
            img.save(
                Path(
                    image_generation_tmp_save_folder,
                    filename,
                )
            )

        # denormalize the images/latents and save to logger if first batch
        # (first batch of main process only to prevent "logger overflow")
        # (and actualy limit to 50 images/latents maximum)
        if batch_idx == 0 and accelerator.is_main_process:
            # log images
            images_processed = (images * 255).round().astype("uint8")  # type: ignore
            accelerator.log(
                {
                    f"generated_samples/{class_name}": [
                        wandb.Image(img) for img in images_processed[:50]
                    ],
                    "epoch": epoch,
                },
                step=global_step,
            )
        progress_bar.update()


@torch.no_grad()
def _compute_log_metrics(
    logger: MultiProcessAdapter,
    class_name: str,
    class_label: int,
    args: Namespace,
    image_generation_tmp_save_folder: Path,
    fidelity_cache_root: Path,
    accelerator: Accelerator,
    global_step: int,
    main_metric_values: list[float],
    raw_dataset: ImageFolder | Subset,
) -> None:
    # log the start of metrics computation
    if args.proba_uncond == 1:
        msg = f"Computing metrics for unconditional generation..."
    else:
        msg = f"Computing metrics for class {class_name}..."
    logger.info(msg)

    # extract a subset of the raw dataset for this class only if applicable
    if args.proba_uncond != 1:
        indices_this_class = np.nonzero(np.array(raw_dataset.targets) == class_label)[0]  # type: ignore
        real_images_this_class = Subset(raw_dataset, list(indices_this_class))
    else:
        real_images_this_class = raw_dataset

    # perform metrics computation
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=image_generation_tmp_save_folder.as_posix(),
        input2=real_images_this_class,
        cuda=True,
        batch_size=args.eval_batch_size,
        isc=args.compute_isc,
        fid=args.compute_fid,
        kid=args.compute_kid,
        verbose=False,
        cache_root=fidelity_cache_root,
        input2_cache_name=f"{class_name}",  # forces caching
        kid_subset_size=args.kid_subset_size,
        samples_find_deep=True,
    )

    for metric_name, metric_value in metrics_dict.items():
        accelerator.log(
            {
                f"{metric_name}/{class_name}": metric_value,
            },
            step=global_step,
        )
        if metric_name == args.main_metric:
            main_metric_values.append(metric_value)

    # clear the tmp folder for this class
    rmtree(image_generation_tmp_save_folder)


def save_pipeline(
    accelerator: Accelerator,
    args: Namespace,
    pipeline: SupportedPipelines,
    full_pipeline_save_folder: Path,
    repo,
    epoch: int,
    logger: MultiProcessAdapter,
    ema_models: dict,
    components_to_train_transcribed: list[str],
    first_save: bool = False,
) -> None:
    if (
        first_save
        and full_pipeline_save_folder.exists()
        and len(list(full_pipeline_save_folder.iterdir())) != 0
    ):
        logger.warn(
            "Not overwriting already populated pipeline save folder at run start"
        )
        return

    # Use the EMA weights if applicable
    pipeline_components = {}
    for module_name, module in pipeline.components.items():
        if module_name in components_to_train_transcribed and args.use_ema:
            # this module is EMA'ed; extract it
            unwrapped_module = accelerator.unwrap_model(module)
            # store its parameters in the EMA model
            ema_models[module_name].store(unwrapped_module.parameters())
            # temporarily copy the EMA parameters to the module
            ema_models[module_name].copy_to(unwrapped_module.parameters())
            # and save that module to the inference pipeline
            pipeline_components[module_name] = accelerator.unwrap_model(
                unwrapped_module
            )
        else:
            pipeline_components[module_name] = accelerator.unwrap_model(module)

    # Create inference pipeline
    match args.model_type:
        case "StableDiffusion":
            inference_pipeline = CustomStableDiffusionImg2ImgPipeline(
                **pipeline_components
            )
        case "DDIM":
            inference_pipeline = ConditionalDDIMPipeline(**pipeline_components)
        case _:
            raise ValueError(f"Unknown model type {args.model_type}")

    # save to full_pipeline_save_folder (≠ initial_pipeline_save_path...)
    logger.info(
        f"Saving full {args.model_type} pipeline to {full_pipeline_save_folder} at epoch {epoch}"
    )
    inference_pipeline.save_pretrained(full_pipeline_save_folder)

    if args.push_to_hub:
        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)


def setup_fine_tuning(args, chckpt_save_path, logger):
    """
    This function copies the pre-trained model's checkpoint to the current chckpt_save_path,
    so that the fine-tuning can be started from the pre-trained model's checkpoint.
    """
    if not chckpt_save_path.endswith("/"):
        chckpt_save_path += "/"

    checkpoint_dir_name = os.path.basename(args.fine_tune_experiment_by_paired_training.rstrip('/'))

    if not os.path.isdir(chckpt_save_path):
        raise NotADirectoryError(f"chckpt_save_path {chckpt_save_path} is not a directory")
    
    if not os.path.isdir(args.fine_tune_experiment_by_paired_training):
        raise NotADirectoryError(f"args.fine_tune_experiment_by_paired_training
                                 {args.fine_tune_experiment_by_paired_training} is not a directory")

    # Creating and running the command
    cmd = f"cp -r {args.fine_tune_experiment_by_paired_training} {chckpt_save_path}"
    exc_code = os.system(cmd)
    if exc_code != 0:
        raise Exception(f"exit code {exc_code} for command: {cmd}")
    logger.info(
        f"Pre-trained checkpoint coppied:\nFrom: {args.fine_tune_experiment_by_paired_training}\nTo: {chckpt_save_path}\n"
    )

    args.resume_from_checkpoint = checkpoint_dir_name
    logger.info(
        f"args.resume_from_checkpoint is updated: {args.resume_from_checkpoint}"
    )


def dice_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    assert pred.shape == target.shape, "pred and target should have the same shape."
    assert torch.all((pred == 0) | (pred == 1)), "pred should have values 0 or 1."
    assert torch.all((target == 0) | (target == 1)), "target should have values 0 or 1."
    
    pred_flat = pred.view(pred.size(0), pred.size(1), -1)
    target_flat = target.view(target.size(0), target.size(1), -1)
    
    intersection = torch.sum(pred_flat * target_flat, dim=2)
    pred_sum = torch.sum(pred_flat, dim=2)
    target_sum = torch.sum(target_flat, dim=2)

    dice = (2 * intersection) / (pred_sum + target_sum + 1e-8)
    return dice.mean().item()


def evaluate_paired_dataset(
    num_update_steps_per_epoch: int,
    accelerator: Accelerator,
    pipeline: ConditionalDDIMPipeline,
    epoch: int,
    dataloader: DataLoader,
    split: str,
    args: Namespace,
    global_step: int,
    lr_scheduler,
    logger: MultiProcessAdapter,
    is_initial_benchmark: bool,
    do_visual_inspection: bool,
   
) -> tuple[int, float | None]:
    
    assert split in ["train", "test"], f"split should be train or test, not {split}"

    def make_tensor_binary(tensor: torch.Tensor) -> torch.Tensor:
        # The input tensor should be centered around 0
        tensor = torch.sigmoid(tensor)
        tensor = torch.round(tensor)
        assert torch.all((tensor == 0) | (tensor == 1)), "tensor should be binary."
        return tensor

    pipe = pipeline
    denoiser_model = pipeline.unet
    denoiser_model.eval()
    num_inference_steps = 30

    dataloader = accelerator.prepare(dataloader)
    # pipe = pipe.to(accelerator.device)
    accelerator.wait_for_everyone()

    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"evaluate_paired_dataset on epoch {epoch}")
    
    # ----------- Iterate Over Batches ----------- #
    mse_all, bce_all, dice_all = [], [], []
    sources_accu = []
    targets_accu = []
    translateds_accu = []
    
    for step, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"evaluate_paired_dataset on {split} set...",
            position=0,
        )
    ):
        
        source_images = batch["source_images"].to(accelerator.device)
        source_labels = batch["source_class"].to(accelerator.device)
        target_images = batch["target_images"].to(accelerator.device)
        target_labels = batch["target_class"].to(accelerator.device)

        orig_class_labels = source_labels
        target_class_labels = 1 - orig_class_labels
        assert torch.equal(target_class_labels, target_labels), "target_class_labels and target_labels should be the same."

        with torch.no_grad():
            translated_images = _ddibـfor_testing_during_training(
                pipe,
                source_images, # [bs, 1, 128, 128]
                orig_class_labels, # [bs] --> [1, 1, 1, 1, 1, 1, ...]
                target_class_labels,
                num_inference_steps,
                process_idx=accelerator.process_index
            )

            if torch.isnan(translated_images).any():
                msg = "NaN values found in translated_images!"
                logger.warn(f"\n{msg}")
                accelerator.log({'warning': msg})
            if torch.isinf(translated_images).any():
                msg = "Inf values found in translated_images!"
                logger.warn(f"\n{msg}")
                accelerator.log({'warning': msg})

            clampped_translated_images = torch.clamp(translated_images, -1, 1)
            mse = F.mse_loss(clampped_translated_images, target_images)
            mse_all.append(mse.cpu().item())
            bce, dice = None, None # Default

            if args.paired_training_loss.lower().strip() == "bce":
                assert all(value in [-1, 1] for value in target_images.unique()), "target_images should be binary in {-1, 1}"
                target_images = make_tensor_binary(target_images) # Going from {-1, 1} to {0, 1}
                num_positive = (target_images == 1).sum().item()
                num_negative = (target_images == 0).sum().item()
                pos_weight = num_negative / (num_positive + 1e-3)
                bce = F.binary_cross_entropy_with_logits(
                    translated_images, target_images, pos_weight=torch.tensor(pos_weight).to(accelerator.device)
                )
                translated_images = make_tensor_binary(translated_images)
                dice = dice_score(pred=translated_images, target=target_images)

                bce_all.append(bce.cpu().item())
                dice_all.append(dice)

            if len(sources_accu) < 20:
                sources_accu.append(source_images)
                targets_accu.append(target_images)
                translateds_accu.append(translated_images)
            
    if is_initial_benchmark:
        logs = {
            f"benchmark_{split}_mse": np.array(mse_all).mean(),
            f"benchmark_{split}_bce": np.array(bce_all).mean(),
            f"benchmark_{split}_dice": np.array(dice_all).mean(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }
    else:
        logs = {
            f"{split}_mse": np.array(mse_all).mean(),
            f"{split}_bce": np.array(bce_all).mean(),
            f"{split}_dice": np.array(dice_all).mean(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }

    accelerator.log(logs, step=global_step)
    accelerator.wait_for_everyone()

    if do_visual_inspection:
        paired_dataset_visual_inspection_and_log(
            accelerator=accelerator,
            global_step=global_step,
            source_images=torch.cat(sources_accu, dim=0),
            target_images=torch.cat(targets_accu, dim=0),
            translated_images=torch.cat(translateds_accu, dim=0),
            split=split
        )
    return logs


class EpochTimer:
    def __init__(self):
        self.start_time = None
        self.start_gs = None
        self.end_time = None
        self.end_gs = None

    def start(self, global_step):
        self.start_time = time.time()
        self.start_gs = global_step
        self.end_time = None
        self.end_gs = None

    def end(self, global_step, accelerator):
        self.end_time = time.time()
        self.end_gs = global_step
        epoch_elapsed_time_mins = round((self.end_time - self.start_time) / 60, 2)
        epoch_elapsed_steps = self.end_gs - self.start_gs
        step_elapsed_time_secs = round((self.end_time - self.start_time) / epoch_elapsed_steps, 2)
        accelerator.log(
            {
                "epoch_elapsed_time_mins": epoch_elapsed_time_mins,
                "step_elapsed_time_secs": step_elapsed_time_secs,
            },
            step=self.end_gs,
        )
        



    

    