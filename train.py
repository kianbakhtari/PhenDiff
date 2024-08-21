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

from argparse import Namespace
from math import inf, sqrt
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

import wandb
from src.args_parser import parse_args
from src.utils_dataset import setup_dataset, setup_paired_dataset
from src.utils_misc import (
    args_checker,
    create_repo_structure,
    get_chckpt_save_path,
    get_HF_component_names,
    get_initial_best_metric,
    modify_args_for_debug,
    setup_logger,
)
from src.utils_models import load_initial_pipeline
from src.utils_training import (
    generate_samples_compute_metrics_save_pipe,
    get_training_setup,
    perform_training_epoch,
    resume_from_checkpoint,
    save_pipeline,
    perform_class_transfer_for_paired_training,
    perform_sample_prediction_for_paired_training,
    setup_fine_tuning,
    evaluate_paired_dataset,
    EpochTimer
)
from src.utils_Img2Img import ClassTransferExperimentParams

logger: MultiProcessAdapter = get_logger(__name__, log_level="INFO")


def main(args: Namespace):
    #####
    visual_inspection_interval = 250 # Also in utils training, needs to be same and set in the args

    # ---------------------------------- Accelerator ---------------------------------
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=Path(
            args.exp_output_dirs_parent_folder, args.experiment_name, args.run_name
        ).as_posix(),
    )

    # unused parameters when unconditionally denoising samples in CLF guidance training for DDIM;
    # not needed for SD as the HF code passes zeros instead of skipping the conditioning part of the network
    # TODO: interesting to think about the implications of these two different methods!
    kwargs_handlers = (
        [DistributedDataParallelKwargs(find_unused_parameters=True)]
        if args.model_type == "DDIM"
        else None
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,  # type: ignore
    )

    # ----------------------------- Setup pre-trained model -----------------------------
    # Setup pre-trained model if fine-tuning with paired dataset
    if accelerator.is_main_process and args.fine_tune_experiment_by_paired_training:
        chckpt_save_path = get_chckpt_save_path(args, accelerator, logger)
        setup_fine_tuning(args, chckpt_save_path, logger)
    accelerator.wait_for_everyone()

    # ----------------------------- Repository Structure -----------------------------
    (
        image_generation_tmp_save_folder,
        initial_pipeline_save_folder,
        full_pipeline_save_folder,
        repo,
        chckpt_save_path,
    ) = create_repo_structure(args, accelerator, logger)
    accelerator.wait_for_everyone()

    fidelity_cache_root: Path = Path(
        args.exp_output_dirs_parent_folder, ".fidelity_cache"
    )

    torch_hub_cache_dir: Path = Path(
        args.exp_output_dirs_parent_folder, ".torch_hub_cache"
    )
    torch.hub.set_dir(torch_hub_cache_dir)

    # ------------------------------------- WandB ------------------------------------
    if accelerator.is_main_process:
        logger.info(
            f"Logging to entity:{args.wandb_entity} | project:{args.experiment_name} | run:{args.run_name}"
        )
        run_id = None
        run_id_file = Path(accelerator_project_config.project_dir, "run_id.txt")
        if run_id_file.exists():
            if args.resume_from_checkpoint is None:
                logger.warning(
                    "Found a 'run_id.txt' file but no 'resume_from_checkpoint' argument was passed; ignoring this file and not resuming W&B run."
                )
            else:
                with open(run_id_file, "r") as f:
                    run_id = f.readline().strip()
                logger.info(
                    f"Found a 'run_id.txt' file; imposing wandb to resume the run with id {run_id}"
                )

        # Init W&B
        init_kwargs = {
            "wandb": {
                "dir": args.exp_output_dirs_parent_folder,
                "name": args.run_name,
                "save_code": True,
                "entity": args.wandb_entity,
            }
        }
        if run_id is not None:
            init_kwargs["wandb"]["id"] = run_id
            init_kwargs["wandb"]["resume"] = "must"

        accelerator.init_trackers(
            project_name=args.experiment_name,
            config=vars(args),
            # save metadata to the "wandb" directory
            # inside the *parent* folder common to all *experiments*
            init_kwargs=init_kwargs,
        )

        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        new_run_id = wandb_tracker.id
        if run_id is not None and new_run_id != run_id:
            logger.warning(
                f"Found a 'run_id.txt' file but the run id in it ({run_id}) is different from the one generated by W&B ({new_run_id}); overwriting the file with the new run id."
            )
        with open(run_id_file, "w+") as f:
            f.write(new_run_id)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    setup_logger(logger, accelerator)

    # ------------------------------------ Checks ------------------------------------
    if accelerator.is_main_process:
        args_checker(args, logger)

    # ------------------------------------ Dataset -----------------------------------
    dataset, raw_dataset, nb_classes = setup_dataset(args, logger)

    num_workers = (
        args.dataloader_num_workers
        if args.dataloader_num_workers is not None
        else accelerator.num_processes
    )

    train_dataloader = torch.utils.data.DataLoader(  # type: ignore
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=args.dataloader_prefetch_factor,
        persistent_workers=args.persistent_workers,
        pin_memory=args.pin_memory
    )

    paired_dataloader = None
    if args.paired_train_data_dir is not None:
        paired_dataset = setup_paired_dataset(args, logger)
        paired_dataloader = torch.utils.data.DataLoader(  # type: ignore
            paired_dataset,
            batch_size=args.paired_train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=args.dataloader_prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory
        )
        
    test_dataloader = None
    if args.test_data_dir is not None:
        test_dataset = setup_paired_dataset(args, logger, test_split=True)
        test_dataloader = torch.utils.data.DataLoader(  # type: ignore
            test_dataset,
            batch_size=args.paired_train_batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=args.dataloader_prefetch_factor,
            persistent_workers=args.persistent_workers,
            pin_memory=args.pin_memory
        )
        
    # ------------------------------------ Debug -------------------------------------
    if args.debug:
        modify_args_for_debug(logger, args, len(train_dataloader))
        if accelerator.is_main_process:
            args_checker(args, logger, False)  # check again after debug modifications 😈

    # --------------------------------- Load Pipeline --------------------------------
    # Download the full (possibly pretrained) pipeline
    # Note that HF pipelines are not meant to used for training;
    # here they are only used as convenient "supermodel" wrappers
    pipeline = load_initial_pipeline(
        args, initial_pipeline_save_folder, logger, nb_classes, accelerator
    ) # Kian: ConditionalDDIMPipeline

    # --------------------------- Move & Freeze Components ---------------------------
    # Move components to device
    pipeline.to(accelerator.device)

    # ❄️ >>> Freeze components <<< ❄️
    if "autoencoder" not in args.components_to_train and hasattr(pipeline, "vae"):
        logger.info(f"Freezing the autoencoder ('vae')")
        pipeline.vae.requires_grad_(False)
    if "denoiser" not in args.components_to_train and hasattr(pipeline, "unet"):
        logger.info(f"Freezing the denoiser ('unet')")
        pipeline.unet.requires_grad_(False)
    if "class_embedding" not in args.components_to_train and hasattr(
        pipeline, "class_embedding"
    ):
        logger.info(f"Freezing the class_embedding ('class_embedding')")
        pipeline.class_embedding.requires_grad_(False)

    # ----------------------------- Attention fine-tuning ----------------------------
    if args.attention_fine_tuning:
        if not hasattr(pipeline, "unet"):
            raise ValueError(
                "Attention fine tuning is only supported for models with a 'unet' attribute"
            )  # a bit artificial, but will probably not be done for vae training anyway
        if "denoiser" not in args.components_to_train and hasattr(pipeline, "unet"):
            raise ValueError(
                "Attention fine tuning requires 'denoiser' to be trained (set --components_to_train)"
            )  # duplicates code, but the args names must convey a meaning!
        logger.info(
            f"--attention_fine_tuning was passed: first freezing the denoiser, then requiring grad on attentions"
        )
        pipeline.unet.requires_grad_(False)
        for module in pipeline.unet.modules():
            if hasattr(module, "attentions"):
                logger.info(
                    f"Found 'attentions' attribute in {module.__class__.__name__}; setting requires_grad to True on these"
                )
                module.attentions.requires_grad_(True)

    # --------------------------------- Miscellaneous --------------------------------
    # Create EMA for the models
    ema_models = {}
    components_to_train_transcribed = get_HF_component_names(args.components_to_train)
    if args.use_ema:
        for module_name, module in pipeline.components.items():
            if module_name in components_to_train_transcribed:
                ema_models[module_name] = EMAModel(
                    module.parameters(),
                    decay=args.ema_max_decay,
                    use_ema_warmup=True,
                    inv_gamma=args.ema_inv_gamma,
                    power=args.ema_power,
                    model_cls=module.__class__,
                    model_config=module.config,
                )
                ema_models[module_name].to(accelerator.device)
        logger.info(
            f"Created EMA weights for the following models: {list(ema_models)} (corresponding to the (unordered) following args: {args.components_to_train})"
        )

    # Track gradients of the denoiser
    if (
        accelerator.is_main_process
        and hasattr(pipeline, "unet")
        and "denoiser" in args.components_to_train
    ):
        wandb.watch(pipeline.unet)

    # ----------------------------- Save Custom Pipeline -----------------------------
    if accelerator.is_main_process:
        save_pipeline(
            accelerator=accelerator,
            args=args,
            pipeline=pipeline,
            full_pipeline_save_folder=full_pipeline_save_folder,
            repo=repo,
            epoch=0,
            logger=logger,
            ema_models=ema_models,
            components_to_train_transcribed=components_to_train_transcribed,
            first_save=True,
        )
    accelerator.wait_for_everyone()

    # ----------------------------------- Optimizer ----------------------------------
    params_to_optimize = []
    for module_name, module in pipeline.components.items():
        if module_name in components_to_train_transcribed:  # was EMA'ed
            params_to_optimize += list(pipeline.components[module_name].parameters())

    # scale the learning rate with the square root of the number of GPUs
    logger.info(
        f"Scaling learning rate with the (square root of the) number of GPUs (×{round(sqrt(accelerator.num_processes), 3)})"
    )
    args.learning_rate *= sqrt(accelerator.num_processes)

    optimizer = torch.optim.AdamW(  # TODO: different params for different components
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---------------------------- Learning Rate Scheduler ----------------------------
    # get the final number of trainig steps
    tot_training_steps: int = min(
        (
            args.max_num_epochs * len(train_dataloader)  # type: ignore
            if args.max_num_epochs is not None
            else inf
        ),
        args.max_num_steps if args.max_num_steps is not None else inf,  # type: ignore
    )

    lr_scheduler = get_scheduler(  # TODO: different params for different components
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=tot_training_steps,
    )

    # ----------------------------- Distributed Compute  -----------------------------
    # get the total len of the dataloader before distributing it
    total_dataloader_len = len(train_dataloader)

    # prepare distributed training with 🤗's magic
    # first all general training helpers
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # then model-specific submodels (is it beyond hackyness acceptability?)
    match args.model_type:
        case "DDIM":
            denoiser = accelerator.prepare(pipeline.unet)
            pipeline.unet = denoiser
        case "StableDiffusion":
            denoiser, autoencoder, class_embedding = accelerator.prepare(
                pipeline.unet, pipeline.vae, pipeline.class_embedding
            )
            pipeline.unet = denoiser
            pipeline.vae = autoencoder
            pipeline.class_embedding = class_embedding
        case _:
            raise ValueError(f"Unknown model type {args.model_type}")

    # -------------------------------- Training Setup --------------------------------
    first_epoch = 0
    global_step = 0
    resume_step = 0

    (
        num_update_steps_per_epoch,
        actual_eval_batch_sizes_for_this_process,
    ) = get_training_setup(
        args,
        accelerator,
        train_dataloader,
        logger,
        list(pipeline.components),
        components_to_train_transcribed,
        len(dataset),
        len(raw_dataset),
        pipeline,
        tot_training_steps,
    )

    # ---------------------------- Resume from Checkpoint ----------------------------
    if args.resume_from_checkpoint:
        first_epoch, resume_step, global_step = resume_from_checkpoint(
            args,
            logger,
            accelerator,
            num_update_steps_per_epoch,
            global_step,
            chckpt_save_path,
        )
    epoch = first_epoch

    # ----------------------------- Initial best metrics -----------------------------
    if accelerator.is_main_process:
        best_metric = get_initial_best_metric()

    # --------------------------------- Getting Ready For Training --------------------------------
    if not args.fine_tune_with_paired_dataset_mode:
        logger.info("Training with unpaired dataset...")
    elif args.fine_tune_with_paired_dataset_mode == "sample":
        logger.info("Training with paired dataset, sample prediction...")
    elif args.fine_tune_with_paired_dataset_mode == "translation":
        logger.info("Training with paired dataset, translation...")
    else:
        raise ValueError(f"args.fine_tune_with_paired_dataset_mode should be either sample or translation,\
                not {args.fine_tune_with_paired_dataset_mode}")
    
    # Some things to do before paired dataset fine-tuning
    if args.fine_tune_with_paired_dataset_mode in ["sample", "translation"]:
        # Reseting the learning rate to the oroiginal value
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=tot_training_steps,
        )
        lr_scheduler = accelerator.prepare(lr_scheduler)

        # Initial evaluation on train dataset
        initial_train_metrics = evaluate_paired_dataset(
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                accelerator=accelerator,
                pipeline=pipeline,
                epoch=epoch,
                dataloader=paired_dataloader,
                split="train",
                args=args,
                global_step=global_step,
                lr_scheduler=lr_scheduler,
                logger=logger,
                is_initial_benchmark=True,
                do_visual_inspection=True,
            )

        # Initial evaluation on test dataset
        initial_test_metrics = evaluate_paired_dataset(
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                accelerator=accelerator,
                pipeline=pipeline,
                epoch=epoch,
                dataloader=test_dataloader,
                split="test",
                args=args,
                global_step=global_step,
                lr_scheduler=lr_scheduler,
                logger=logger,
                is_initial_benchmark=True,
                do_visual_inspection=True,
            )
        
        epoch_timer = EpochTimer()

    # --------------------------------- Training loop -------------------------------- 
    last_global_step_of_test_visual_inspection = 0  
    while (
        epoch < (args.max_num_epochs if args.max_num_epochs is not None else inf)
        and global_step < tot_training_steps
    ):
        if not args.fine_tune_with_paired_dataset_mode:

            global_step, best_metric = perform_training_epoch(
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                accelerator=accelerator,
                pipeline=pipeline,
                ema_models=ema_models,
                components_to_train_transcribed=components_to_train_transcribed,
                epoch=epoch,
                train_dataloader=train_dataloader,
                args=args,
                first_epoch=first_epoch,
                resume_step=resume_step,
                global_step=global_step,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                logger=logger,
                params_to_clip=params_to_optimize,
                tot_training_steps=tot_training_steps,
                image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                fidelity_cache_root=fidelity_cache_root,
                actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                nb_classes=nb_classes,
                dataset=dataset,
                raw_dataset=raw_dataset,
                full_pipeline_save_folder=full_pipeline_save_folder,
                repo=repo,
                best_metric=best_metric if accelerator.is_main_process else None,  # type: ignore
                chckpt_save_path=chckpt_save_path,
                paired_dataloader=paired_dataloader
            )

        elif args.fine_tune_with_paired_dataset_mode == "sample":

            global_step, best_metric, loss_value = perform_sample_prediction_for_paired_training(
                    num_update_steps_per_epoch=num_update_steps_per_epoch,
                    accelerator=accelerator,
                    pipeline=pipeline,
                    ema_models=ema_models,
                    components_to_train_transcribed=components_to_train_transcribed,
                    epoch=epoch,
                    dataloader=paired_dataloader,
                    args=args,
                    first_epoch=first_epoch,
                    resume_step=resume_step,
                    global_step=global_step,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    logger=logger,
                    params_to_clip=params_to_optimize,
                    tot_training_steps=tot_training_steps,
                    image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                    fidelity_cache_root=fidelity_cache_root,
                    actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                    nb_classes=nb_classes,
                    dataset=dataset,
                    raw_dataset=raw_dataset,
                    full_pipeline_save_folder=full_pipeline_save_folder,
                    repo=repo,
                    best_metric=best_metric if accelerator.is_main_process else None,  # type: ignore
                    chckpt_save_path=chckpt_save_path,
                )

        elif args.fine_tune_with_paired_dataset_mode == "translation":
            epoch_timer.start(global_step)
            global_step, best_metric, loss_value = perform_class_transfer_for_paired_training(
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                accelerator=accelerator,
                pipeline=pipeline,
                ema_models=ema_models,
                components_to_train_transcribed=components_to_train_transcribed,
                epoch=epoch,
                dataloader=paired_dataloader,
                args=args,
                first_epoch=first_epoch,
                resume_step=resume_step,
                global_step=global_step,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                logger=logger,
                params_to_clip=params_to_optimize,
                tot_training_steps=tot_training_steps,
                image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                fidelity_cache_root=fidelity_cache_root,
                actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                nb_classes=nb_classes,
                full_pipeline_save_folder=full_pipeline_save_folder,
                repo=repo,
                best_metric=best_metric if accelerator.is_main_process else None,  # type: ignore
                chckpt_save_path=chckpt_save_path,
            )
            epoch_timer.end(global_step, accelerator)

        if not args.fine_tune_with_paired_dataset_mode:
        # Generate sample images for visual inspection & metrics computation
            if args.eval_save_model_every_epochs is not None and (
                epoch % args.eval_save_model_every_epochs == 0
                or (
                    args.precise_first_n_epochs is not None
                    and epoch < args.precise_first_n_epochs
                )
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
                    best_metric if accelerator.is_main_process else None,  # type: ignore
                    full_pipeline_save_folder,
                    repo,
                )
        else:
            logger.info("Evaluation on test set...")
            do_visual_inspection = True if global_step - last_global_step_of_test_visual_inspection > visual_inspection_interval else False
            if do_visual_inspection:
                last_global_step_of_test_visual_inspection = global_step
            test_metrics = evaluate_paired_dataset(
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                accelerator=accelerator,
                pipeline=pipeline,
                epoch=epoch,
                dataloader=test_dataloader,
                split="test",
                args=args,
                global_step=global_step,
                lr_scheduler=lr_scheduler,
                logger=logger,
                is_initial_benchmark=False,
                do_visual_inspection=do_visual_inspection,
            )

        # do not start new epoch before generation & pipeline saving is done
        accelerator.wait_for_everyone()
        epoch += 1

    accelerator.end_training()


if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args)
