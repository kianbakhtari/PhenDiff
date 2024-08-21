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

import os
import shutil
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_fidelity
import yaml
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from datasets import load_dataset
from diffusers.schedulers import DDIMInverseScheduler
from omegaconf import DictConfig
from PIL import Image
from scipy.stats import normaltest
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import wandb

from .custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)
from .pipeline_conditional_ddim import ConditionalDDIMPipeline

DEBUG_BATCHES_LIMIT = 0
MAX_NB_LOGGED_IMAGES = 50
DEBUG_KID_SUBSET_SIZE = 1


class ClassTransferExperimentParams:
    def __init__(
        self,
        class_transfer_method: str,
        pipes: dict,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset],
        cfg: DictConfig,
        output_dir: str,
        logger: MultiProcessAdapter,
        accelerator: Accelerator,
        num_inference_steps: Optional[int],
        dataset_name: str,
        fidelity_cache_root: str,
    ):
        self.class_transfer_method = class_transfer_method
        self.pipes = pipes
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.cfg = cfg
        self.output_dir = output_dir
        self.logger = logger
        self.accelerator = accelerator
        self.num_inference_steps = num_inference_steps
        self.dataset_name = dataset_name
        self.fidelity_cache_root = fidelity_cache_root


@torch.no_grad()
def check_Gaussianity(gauss: Tensor) -> None:
    print(
        f"Checking Gausianity of components of tensor of shape {tuple(gauss.shape)}..."
    )
    _, axes = plt.subplots(nrows=1, ncols=len(gauss), figsize=(16, 3))
    for idx, itm in enumerate(gauss):
        axes[idx].hist(itm.cpu().numpy().flatten(), bins=100, range=(-3, 3))
        print(
            f"Gaussian(?) {idx}: mean={itm.mean().item()}, std={itm.std().item()}",
            end="; ",
        )
        _, p = normaltest(itm.cpu().numpy().flatten())
        print(f"2-sided Χ² probability for the normality hypothesis: {p}")
    plt.show()


@torch.no_grad()
def tensor_to_PIL(
    tensor: Tensor,
    channel: int | str = "mean",
) -> Image.Image | list[Image.Image]:
    assert tensor.ndim == 4, "Expecting a tensor of shape (N, C, H, W)"
    assert channel in ["mean"] + list(
        range(tensor.shape[1])
    ), f"Expecting a channel in {list(range(tensor.shape[1]))} or 'mean', got {channel}"

    img_to_show = tensor.clone().detach()
    # these are latent vectors => return a grayscale image
    if tensor.shape[1] == 4:
        # min-max normalization
        img_to_show -= img_to_show.min()
        img_to_show /= img_to_show.max()
        img_to_show = img_to_show.clamp(0, 1)
        if isinstance(channel, int):
            # take the given channel only for visualization
            img_to_show = img_to_show[:, channel].view(
                tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]
            )
        elif channel == "mean":
            # convert to grayscale taking the mean over the 4 channels
            img_to_show = img_to_show.mean(dim=1, keepdim=True)
    # these are "true" images => return a color image
    elif tensor.shape[1] == 3:
        assert (
            tensor.min() >= -1 and tensor.max() <= 1
        ), f"Expecting values in [-1, 1], got tensor.min()={tensor.min()} and tensor.max()={tensor.max()}"
        if tensor.min() != -1:
            print(
                f"Warning in tensor_to_PIL: tensor.min() = {tensor.min().item()} != -1"
            )
        if tensor.max() != 1:
            print(
                f"Warning in tensor_to_PIL: tensor.max() = {tensor.max().item()} != -1"
            )
        img_to_show = (img_to_show / 2 + 0.5).clamp(0, 1)

    # convert to PIL image
    img_to_show = img_to_show.cpu().permute(0, 2, 3, 1).numpy()
    img_to_show = (img_to_show * 255).round().astype("uint8")
    if img_to_show.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            Image.fromarray(image.squeeze(), mode="L") for image in img_to_show
        ]
    else:
        pil_images = [Image.fromarray(image) for image in img_to_show]
    # if only one image return it directly
    if len(pil_images) == 1:
        return pil_images[0]

    return pil_images


def print_grid(
    list_PIL_images: list[Image.Image],
    nb_img_per_row: int = 5,
    titles: list[str] | None = None,
    figsize=(16, 4),
) -> None:
    if titles is not None:
        assert len(titles) == len(list_PIL_images)
    num_images = len(list_PIL_images)
    nrows = ceil(num_images / nb_img_per_row)
    _, axes = plt.subplots(
        nrows=nrows, ncols=nb_img_per_row, figsize=(figsize[0], figsize[1] * nrows)
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i in range(num_images):
        row_nb = i // nb_img_per_row
        col_nb = i % nb_img_per_row
        axes[row_nb, col_nb].imshow(list_PIL_images[i].convert("RGB"))
        axes[row_nb, col_nb].axis("off")
        if titles is not None:
            axes[row_nb, col_nb].set_title(titles[i], fontsize=10)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def hack_class_embedding(cl_embed: Tensor) -> Tensor:
    """Hack to match the expected encoder_hidden_states shape"""
    assert cl_embed.ndim == 2, "Expecting a tensor of shape (N, E)"
    (bs, ed) = cl_embed.shape
    cl_embed = cl_embed.reshape(bs, 1, ed)
    padding = torch.zeros_like(cl_embed).repeat(1, 76, 1).to(cl_embed.device)
    cl_embed = torch.cat([cl_embed, padding], dim=1)
    return cl_embed


@torch.no_grad()
def load_datasets(
    cfg: DictConfig, dataset_name: str
) -> Tuple[Dataset, Optional[Dataset]]:
    """TODO: docstring"""
    # data preprocessing
    normalize = cfg.dataset[dataset_name].normalize
    preproc = transforms.Compose(
        [
            transforms.Resize(
                cfg.dataset[dataset_name].definition,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([normalize[0]], [normalize[1]]),
        ]
    )

    def transform_images(examples):
        images = [preproc(image.convert("RGB")) for image in examples["image"]]
        file_basenames = [
            os.path.splitext(os.path.basename(image.filename))[0]
            for image in examples["image"]
        ]
        class_labels = torch.tensor(examples["label"]).long()
        return {
            "images": images,
            "file_basenames": file_basenames,
            "class_labels": class_labels,
        }

    # instantiate datasets
    train_dataset: Dataset = load_dataset(  # type: ignore
        "imagefolder",
        data_dir=cfg.dataset[dataset_name].root,
        split="train",
    )
    if Path(cfg.dataset[dataset_name].root, "test").exists():
        test_dataset = load_dataset(  # type: ignore
            "imagefolder",
            data_dir=cfg.dataset[dataset_name].root,
            split="test",
        )
        test_dataset.set_transform(transform_images)  # type: ignore
        test_dataset.classes = test_dataset.features["label"].names  # type: ignore
    else:
        test_dataset = None

    train_dataset.set_transform(transform_images)  # type: ignore
    # mimic PyTorch ImageFolder
    train_dataset.classes = train_dataset.features["label"].names  # type: ignore

    return train_dataset, test_dataset  # type: ignore


def Lp_loss(
    x: torch.Tensor, y: torch.Tensor, p: int | float | Literal["inf", "-inf"] = 2
) -> torch.Tensor:
    """Returns the L_p norms of the flattened `(x[i] - y[i])` or `(x[i] - y)` vectors for each `i` in the batch.

    Arguments
    ---------
    - x: `torch.Tensor`, shape `(N, C, H, W)`
    - y: `torch.Tensor`, shape `(C, H, W)` or `(N, C, H, W)`
    - p: `int | float | "inf" | "-inf"`; default to `2`

    Returns
    -------
    `torch.linalg.vector_norm(x - y, dim=(1, 2, 3), ord=p)`, that is:
    ```
        torch.linalg.vector_norm(x[i] - y, ord=p) for i in range(N)
    ```
    """
    assert (
        x.shape[1:] == y.shape or x.shape == y.shape
    ), f"y.shape={y.shape} should be equal to either x.shape or x.shape[1:] (x.shape={x.shape})"
    assert len(y.shape) in (
        3,
        4,
    ), f"y.shape={y.shape} should be (C, H, W) or (N, C, H, W)"
    return torch.linalg.vector_norm(x - y, dim=(1, 2, 3), ord=p)


def perform_class_transfer_experiment(args: ClassTransferExperimentParams):
    """
    Generate and save to disk the translated images for the specified class transfer method.

    - Iterate first over the pipelines then over the dataset splits and finally on the actual data.

    - Each translated batch is immediately saved to disk and the save directory structure is:

    ```
    output_dir/class_transfer_method/pipeline_name/split/target_class_name
    ```
    where `target_class_name` is the name of the class to which the images have been transferred.

    - Images are further saved with the following naming convention:

    ```
    <original_file_basename>_to_<target_class_index>.png
    ```
    """
    for pipename, pipe in args.pipes.items():
        '''
        Kian.
        pipe is:
        ConditionalDDIMPipeline {
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
        '''
        # ----------- Get the number of inference steps ----------- #
        if args.num_inference_steps is None:
            num_inference_steps = args.cfg.pipeline[pipename].num_inference_steps
        else:
            num_inference_steps = args.num_inference_steps

        # ----------- Creat data loader from dataset ----------- #
        for split, dataset in zip(
            ["train", "test"], [args.train_dataset, args.test_dataset]
        ):
            # There might not be a test dataset
            if dataset is None:
                continue

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(  # type: ignore
                dataset,
                batch_size=args.cfg.batch_size[pipename][
                    f"{args.class_transfer_method}"
                ],
                shuffle=True,
            )

            # ----------- Distributed inference & device placement ----------- #
            dataloader = args.accelerator.prepare(dataloader)
            pipe.unet = args.accelerator.prepare_model(pipe.unet, evaluation_mode=True)
            # pipe = pipe.to(args.accelerator.device)

            # ----------- Creat save directories ----------- #
            save_dir = (
                args.output_dir + f"/{args.class_transfer_method}/{pipename}/{split}"
            )
            if args.accelerator.is_main_process:
                per_target_class_dirs = [
                    save_dir + f"/{class_name}" for class_name in dataset.classes  # type: ignore
                ]
                for per_target_class_dir in per_target_class_dirs:
                    os.makedirs(per_target_class_dir)

            args.accelerator.wait_for_everyone()

            # ----------- Iterate Over Batches ----------- #
            for step, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Running {pipename} on {split} dataset",
                    position=0,
                )
            ):
                # Get batch
                clean_images = batch["images"].to(args.accelerator.device) # Kian: [bs, 3, 128, 128]
                orig_class_labels = batch["class_labels"].to(args.accelerator.device) # Kian: tensor([1, 0, 1, 1, 0, 1, ...])
                # only works for the binary case...
                target_class_labels = 1 - orig_class_labels
                filenames = batch["file_basenames"]

                match args.class_transfer_method:
                    case "linear_interp_custom_guidance_inverted_start":
                        images_to_save = _linear_interp_custom_guidance_inverted_start(
                            pipe,
                            clean_images,
                            orig_class_labels,
                            target_class_labels,
                            args.cfg,
                            num_inference_steps,
                        )
                    case "classifier_free_guidance_forward_start":
                        images_to_save = _classifier_free_guidance_forward_start(
                            pipe,
                            clean_images,
                            target_class_labels,
                            args.cfg,
                            num_inference_steps,
                        )
                    case "ddib":
                        images_to_save = _ddib(
                            pipe,
                            clean_images,
                            orig_class_labels,
                            target_class_labels,
                            num_inference_steps,
                            process_idx=args.accelerator.process_index,
                        )
                        # Kian: images_to_save had shape [bs, ch, H, W] in pipeline but permuted into [bs, H, W, ch] before being returned to here

                    case "inverted_regeneration":
                        # inverted regen is just DDIB with the original class as target!
                        target_class_labels = orig_class_labels
                        images_to_save = _ddib(
                            pipe,
                            clean_images,
                            orig_class_labels,
                            target_class_labels,
                            num_inference_steps,
                            process_idx=args.accelerator.process_index,
                        )
                    case _:
                        raise ValueError(
                            f"Unknown class transfer method: {args.class_transfer_method}"
                        )

                # Save images to disk
                # Kian: images_to_save are saved and logged with [bs, H, W, ch]
                if isinstance(images_to_save, Image.Image):
                    images_to_save = [images_to_save]
                for i, image_to_save in enumerate(images_to_save):
                    target_class_name = dataset.classes[target_class_labels[i]]  # type: ignore
                    save_filename = (
                        save_dir
                        + f"/{target_class_name}"
                        + f"/{filenames[i]}_to_{target_class_name}.png"
                    )
                    image_to_save.save(save_filename)  # type: ignore

                # Log the first NB_LOGGED_IMAGES images (orig & transferred) to WandB
                if step == 0 and args.accelerator.is_main_process:
                    # original samples
                    clean_images_processed_for_logging = (
                        clean_images.detach()[:MAX_NB_LOGGED_IMAGES]
                        .permute(0, 2, 3, 1) # Kian: moving ch to end: [bs, ch, H, W] -> [bs, H, W, ch]
                        .cpu()
                        .numpy()
                    )
                    orig_samples_to_log = [
                        wandb.Image(
                            clean_images_processed_for_logging[i],
                            caption=filenames[i],
                        )
                        for i in range(len(clean_images_processed_for_logging))
                    ]
                    # transferred samples
                    transferred_samples_to_log = [
                        wandb.Image(
                            images_to_save[i], # Kian: shape is already [bs, H, W, ch]
                            caption=f"{filenames[i]}_to_{dataset.classes[target_class_labels[i]]}",  # type: ignore
                        )
                        for i in range(len(clean_images_processed_for_logging))
                    ]
                    data_to_log = [
                        [
                            orig_img_name,
                            orig_img,
                            transf_img,
                            transf_img._caption,  # type: ignore
                        ]
                        for orig_img_name, orig_img, transf_img in zip(
                            filenames[:MAX_NB_LOGGED_IMAGES],
                            orig_samples_to_log,
                            transferred_samples_to_log,
                        )
                    ]
                    table = wandb.Table(
                        [
                            "Original file name",
                            "Original sample",
                            "Transferred sample",
                            "Transferred file name",
                        ],
                        data_to_log,
                    )
                    args.accelerator.log(
                        {
                            f"{args.class_transfer_method}/{pipename}/{split}/samples": table
                        }
                    )

                # Stop if debug flag
                if args.cfg.debug and step >= DEBUG_BATCHES_LIMIT:
                    args.logger.warn(
                        f"debug flag: stopping after {DEBUG_BATCHES_LIMIT+1} batche(s), transferred {clean_images.shape[0]} images"
                    )
                    break


def compute_metrics(
    args: ClassTransferExperimentParams,
):
    for pipename in args.pipes:
        for split, dataset in zip(
            ["train", "test"], [args.cfg.dataset[args.dataset_name].root] * 2
        ):
            # -1 Check if split exists
            if not Path(dataset, split).exists():
                continue

            # 0. Misc.
            bs = args.cfg.batch_size[pipename][args.class_transfer_method]

            nb_samples = len(list(Path(dataset, split).iterdir()))
            if nb_samples < args.cfg.kid_subset_size:
                compute_kid = False
            elif args.cfg.compute_kid:
                compute_kid = True
            else:
                compute_kid = False

            # 1. Unconditional
            args.logger.info("Computing metrics (unconditional case)")
            # get the images to compare
            true_images = Path(dataset, split).as_posix()
            generated_images = (
                args.output_dir + f"/{args.class_transfer_method}/{pipename}/{split}"
            )
            # compute metrics
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=generated_images,
                input2=true_images,
                cuda=True,
                batch_size=bs * 4,
                isc=args.cfg.compute_isc,
                fid=args.cfg.compute_fid,
                kid=compute_kid,
                verbose=False,
                cache_root=args.fidelity_cache_root,
                # input2_cache_name=f"{class_name}",  # TODO: force caching
                kid_subset_size=args.cfg.kid_subset_size,
                samples_find_deep=True,
            )
            # prepare to log metrics
            args.logger.info(
                f"Unconditional metrics for {args.class_transfer_method} with {pipename} on {split} split: {metrics_dict}"
            )
            table = wandb.Table(columns=["Metric", "Value"])
            for metric_name, metric_value in metrics_dict.items():
                table.add_data("uncond " + metric_name, metric_value)

            # 2. Conditional
            args.logger.info("Computing metrics (conditional case)")
            # get the classes
            classes = args.train_dataset.classes  # type: ignore
            assert (
                args.test_dataset is None
                or classes == args.test_dataset.classes  # type: ignore
            ), "Expecting the same classes between train and test datasets"
            # compute metrics per-class
            for class_name in classes:
                # get the images to compare
                true_images = Path(dataset, split, class_name).as_posix()
                generated_images = (
                    args.output_dir
                    + f"/{args.class_transfer_method}/{pipename}/{split}/{class_name}"
                )  #                                                      ^^^^^^^^^^ this is the target class
                # compute metrics
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=generated_images,
                    input2=true_images,
                    cuda=True,
                    batch_size=bs * 4,
                    isc=args.cfg.compute_isc,
                    fid=args.cfg.compute_fid,
                    kid=compute_kid,
                    verbose=False,
                    cache_root=args.fidelity_cache_root,
                    # input2_cache_name=f"{class_name}",  # TODO: force caching
                    kid_subset_size=args.cfg.kid_subset_size,
                    samples_find_deep=False,
                )
                # log metrics
                args.logger.info(
                    f"Conditional metrics for {args.class_transfer_method} with {pipename} on {split} split & target class {class_name}: {metrics_dict}"
                )
                for metric_name, metric_value in metrics_dict.items():
                    table.add_data(class_name + " " + metric_name, metric_value)
            args.accelerator.log(
                {f"{args.class_transfer_method}/{pipename}/{split}/metrics": table}
            )

            if args.cfg.sweep_metric is not None:
                [mthd, p, s, sweep_metric] = args.cfg.sweep_metric.split("/")
                if args.class_transfer_method == mthd and pipename == p and split == s:
                    metrics_df = table.get_dataframe().set_index("Metric")
                    value = metrics_df.at[sweep_metric, "Value"]
                    args.logger.info(
                        f"Logging sweep metric with name '{args.cfg.sweep_metric}' and value {value}"
                    )
                    args.accelerator.log({f"{args.cfg.sweep_metric}": value})


@torch.no_grad()
def _ddib(
    pipe,
    clean_images: Tensor,
    orig_class_labels: Tensor,
    target_class_labels: Tensor,
    num_inference_steps: int,
    process_idx: int,
):
    """TODO: docstring"""
    # Preprocess inputs if LDM
    # target_class_labels must be saved for filename
    if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        clean_images, [orig_class_cond] = _LDM_preprocess(  # type: ignore
            pipe, clean_images, [orig_class_labels]
        )
    else:
        orig_class_cond = orig_class_labels

    # Perform inversion
    inverted_gauss = _inversion(
        pipe, clean_images, orig_class_cond, num_inference_steps, process_idx
    )

    # Perform generation
    if isinstance(pipe, ConditionalDDIMPipeline):
        images_to_save = pipe(
            for_testing=True,
            class_labels=target_class_labels,
            w=0,
            num_inference_steps=num_inference_steps,
            start_image=inverted_gauss,
            add_forward_noise_to_image=False,
            frac_diffusion_skipped=0,
        ).images  # type: ignore
    elif isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        images_to_save = pipe(
            image=inverted_gauss,
            class_labels=target_class_labels,
            strength=1,
            add_forward_noise_to_image=False,
            num_inference_steps=num_inference_steps,
            guidance_scale=0,  # guidance_scale <= 1.0 disables guidance
        )
    else:
        raise NotImplementedError

    return images_to_save


def _ddib_for_training(
    pipe,
    clean_images: Tensor,
    orig_class_labels: Tensor,
    target_class_labels: Tensor,
    num_inference_steps: int,
    process_idx: int,
):
    """TODO: docstring"""
    # Preprocess inputs if LDM
    # target_class_labels must be saved for filename

    orig_class_cond = orig_class_labels

    # Perform inversion
    inverted_gauss = _inversion_for_training(
        pipe, clean_images, orig_class_cond, num_inference_steps, process_idx
    )

    # Perform generation
    if not isinstance(pipe, ConditionalDDIMPipeline):
        raise NotImplementedError
    
    images_to_save = pipe(
        for_testing=False,
        class_labels=target_class_labels,
        w=0,
        num_inference_steps=num_inference_steps,
        start_image=inverted_gauss,
        add_forward_noise_to_image=False,
        frac_diffusion_skipped=0,
    )

    return images_to_save

@torch.no_grad()
def _ddibـfor_testing_during_training(
    pipe,
    clean_images: Tensor,
    orig_class_labels: Tensor,
    target_class_labels: Tensor,
    num_inference_steps: int,
    process_idx: int,
):
    """TODO: docstring"""
    # Preprocess inputs if LDM
    # target_class_labels must be saved for filename
    if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        clean_images, [orig_class_cond] = _LDM_preprocess(  # type: ignore
            pipe, clean_images, [orig_class_labels]
        )
    else:
        orig_class_cond = orig_class_labels

    # Perform inversion
    inverted_gauss = _inversion(
        pipe, clean_images, orig_class_cond, num_inference_steps, process_idx
    )

    # Perform generation
    if isinstance(pipe, ConditionalDDIMPipeline):
        images_to_save = pipe(
            for_testing=True,
            class_labels=target_class_labels,
            w=0,
            num_inference_steps=num_inference_steps,
            start_image=inverted_gauss,
            add_forward_noise_to_image=False,
            frac_diffusion_skipped=0,
        )
    else:
        raise NotImplementedError

    return images_to_save


@torch.no_grad()
def _classifier_free_guidance_forward_start(
    pipe,
    clean_images: Tensor,
    target_class_labels: Tensor,
    cfg: DictConfig,
    num_inference_steps: int,
) -> Image.Image | list[Image.Image]:
    """TODO: docstring"""
    # Perform guided generation
    this_exp_cfg = cfg.class_transfer_method.classifier_free_guidance_forward_start
    guidance_scale = this_exp_cfg.guidance_scale
    frac_diffusion_skipped = this_exp_cfg.frac_diffusion_skipped

    if isinstance(pipe, ConditionalDDIMPipeline):
        images_to_save = pipe(
            class_labels=target_class_labels,
            w=guidance_scale,
            num_inference_steps=num_inference_steps,
            start_image=clean_images,
            frac_diffusion_skipped=frac_diffusion_skipped,
        ).images  # type: ignore
    elif isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        images_to_save = pipe(
            image=clean_images,
            class_labels=target_class_labels,
            strength=frac_diffusion_skipped,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
    else:
        raise NotImplementedError

    return images_to_save  # type: ignore


def _linear_interp_custom_guidance_inverted_start(
    pipe,
    clean_images: Tensor,
    orig_class_labels: Tensor,
    target_class_labels: Tensor,
    cfg: DictConfig,
    num_inference_steps: int,
) -> Image.Image | list[Image.Image]:
    """TODO: docstring"""
    # Preprocess inputs if LDM
    # (target_class_labels must be saved for filename)
    target_class_embeds = None
    if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        clean_images, (
            orig_class_labels,
            target_class_embeds,
        ) = _LDM_preprocess(  # type: ignore
            pipe, clean_images, [orig_class_labels, target_class_labels]
        )

    # Perform inversion
    inverted_gauss = _inversion(
        pipe, clean_images, orig_class_labels, num_inference_steps, None
    )

    # Perform guided generation
    target_cond = (
        target_class_embeds if target_class_embeds is not None else target_class_labels
    )
    image = _custom_guided_generation(
        pipe,
        inverted_gauss,
        target_cond,
        cfg,
        num_inference_steps,
    )

    # Decode from latent space if LDM
    if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
        image = _decode_to_images(pipe, image)
        # also normalize image back to [-1, 1]
        image -= image.min()
        image /= image.max()
        image = image * 2 - 1

    images_to_save = tensor_to_PIL(image)

    return images_to_save


def _custom_guided_generation(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    input_images: Tensor,
    target_class_labels: Tensor,
    cfg: DictConfig,
    num_inference_steps: int,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    images = input_images.clone().detach()

    # setup the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps)

    # perform guided generation
    for t in pipe.scheduler.timesteps:
        # 0. reset grads on images
        images = images.detach().requires_grad_()

        # 1. predict noise/velocity/etc.
        model_output = pipe.unet(
            images,
            t,
            target_class_labels,
        ).sample

        # 2. get x_0 prediction
        x0 = pipe.scheduler.step(
            model_output,
            t,
            images,
        ).pred_original_sample

        # 3. compute loss
        # each image in batch has its own loss with respect to the original sample
        # hence losses.shape = (batch_size,)
        losses = Lp_loss(
            x0,
            input_images,
            cfg.class_transfer_method.linear_interp_custom_guidance_inverted_start.p,
        )

        # 4. get gradient
        losses_seq = [losses[i] for i in range(len(input_images))]
        guidance_grad = torch.autograd.grad(losses_seq, images)[0]

        # 5. modify the image based on this gradient
        guidance_loss_scale = (
            cfg.class_transfer_method.linear_interp_custom_guidance_inverted_start.guidance_loss_scale
        )
        images = images.detach() - guidance_loss_scale * guidance_grad

        # 6. x_t -> x_t-1
        images = pipe.scheduler.step(
            model_output,
            t,
            images,
        ).prev_sample

    return images


@torch.no_grad()
def _inversion(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    input_images: Tensor,
    class_labels: Tensor,
    num_inference_steps: int,
    proc_idx: int | None,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    gauss = input_images.clone().detach()

    # setup the inverted DDIM scheduler
    DDIM_inv_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(  # type: ignore
        pipe.scheduler.config,
    )
    DDIM_inv_scheduler.set_timesteps(num_inference_steps)

    # invert the diffeq
    # for t in tqdm(
    #     DDIM_inv_scheduler.timesteps,
    #     position=proc_idx + 1 if proc_idx is not None else 0,
    #     desc=(
    #         "Inverting images" + f" on process {proc_idx}"
    #         if proc_idx is not None
    #         else ""
    #     ),
    #     leave=False,
    # ):

    for t in DDIM_inv_scheduler.timesteps:
        model_output = pipe.unet(gauss, t, class_labels).sample

        gauss = DDIM_inv_scheduler.step(
            model_output,
            t,  # type: ignore
            gauss,  # type: ignore
        ).prev_sample  # type: ignore

    return gauss


def _inversion_for_training(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    input_images: Tensor,
    class_labels: Tensor,
    num_inference_steps: int,
    proc_idx: int | None,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    # gauss = input_images.clone().detach()
    gauss = input_images

    # setup the inverted DDIM scheduler
    DDIM_inv_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(  # type: ignore
        pipe.scheduler.config,
    )
    DDIM_inv_scheduler.set_timesteps(num_inference_steps)

    # invert the diffeq
    # for t in tqdm(
    #     DDIM_inv_scheduler.timesteps,
    #     position=proc_idx + 1 if proc_idx is not None else 0,
    #     desc=(
    #         "Inverting images" + f" on process {proc_idx}"
    #         if proc_idx is not None
    #         else ""
    #     ),
    #     leave=False,
    # ):
    
    for t in DDIM_inv_scheduler.timesteps:
        model_output = pipe.unet(gauss, t, class_labels).sample

        gauss = DDIM_inv_scheduler.step(
            model_output,
            t,  # type: ignore
            gauss,  # type: ignore
        ).prev_sample  # type: ignore

    return gauss



@torch.no_grad()
def _LDM_preprocess(
    pipe: CustomStableDiffusionImg2ImgPipeline,
    images: Tensor,
    class_labels_seq: Optional[list[Tensor]] = None,
) -> tuple[Tensor, Optional[list[Tensor]]]:
    # encode images to latent space...
    latents = _encode_to_latents(pipe, images)
    # ...and class labels to class embedding
    if class_labels_seq is None:
        class_labels_embeds_seq = None
    else:
        class_labels_embeds_seq = []
        for class_labels in class_labels_seq:
            class_labels_embeds = pipe._encode_class(
                class_labels=class_labels,
                device=class_labels.device,
                do_classifier_free_guidance=False,
            )
            # hack to match the expected encoder_hidden_states shape
            class_labels_embeds = hack_class_embedding(class_labels_embeds)
            class_labels_embeds_seq.append(class_labels_embeds)
    return latents, class_labels_embeds_seq


@torch.no_grad()
def _encode_to_latents(
    pipe: CustomStableDiffusionImg2ImgPipeline, gauss: Tensor
) -> Tensor:
    # encode
    latent = pipe.vae.encode(gauss).latent_dist.sample()
    # scale
    latent *= pipe.vae.config.scaling_factor
    return latent


@torch.no_grad()
def _decode_to_images(
    pipe: CustomStableDiffusionImg2ImgPipeline, latents: Tensor
) -> Tensor:
    # unscale
    latents /= pipe.vae.config.scaling_factor
    # decode
    images = pipe.vae.decode(latents, return_dict=False)[0]
    return images


def modify_debug_args(
    cfg: DictConfig, logger: MultiProcessAdapter
) -> Tuple[Optional[int], DictConfig]:
    # Quick gen if debug flag
    if cfg.debug:
        num_inference_steps = 10
        logger.warning(
            f"Debug mode: setting num_inference_steps to {num_inference_steps} and kid_subset_size to {DEBUG_KID_SUBSET_SIZE}"
        )
        cfg.kid_subset_size = DEBUG_KID_SUBSET_SIZE
    else:  # else let each pipeline have its own param
        num_inference_steps = None

    return num_inference_steps, cfg


def _get_config_path_and_name(cfg: DictConfig, hydra_cfg) -> tuple[Path, Path]:
    """Returns the 'config_path' and 'config_name' args passed to hydra.

    Arguments
    =========
    - cfg: `DictConfig`

        The config object passed to the main function by hydra.

    - hydra_cfg

        The hydra-specific config automatically retrieved by hydra (typically accessed with `hydra.core.hydra_config.HydraConfig.get()`)

    Returns
    =======
    - config_path: `Path`

        The search path where hydra has searched the config file used to launch the job.

    - config_name: `Path`

        The name of the config file used to launch the job.
    """
    # 1. Get the config_path (passed as config_path to hydra)
    launcher_config_path = None
    for cfg_source in hydra_cfg.runtime.config_sources:
        if cfg_source.provider == "main":
            assert (  # just a quick check
                cfg_source.schema == "file"
            ), "Internal error: Expected 'file://' schema from 'main' config source"
            launcher_config_path = Path(cfg_source.path)
            break
    if launcher_config_path is None:
        raise RuntimeError("Could not find main config path")

    # 2. Get the config name
    task_config_name = Path(hydra_cfg.job.config_name)

    return launcher_config_path, task_config_name


def duplicate_config_to_experiment_folder(cfg: DictConfig, hydra_cfg):
    # 1. Get this launcher's config
    launcher_config_path, launcher_config_name = _get_config_path_and_name(
        cfg, hydra_cfg
    )

    # 2. Get the experiment folder
    this_experiment_folder = Path(cfg.exp_parent_folder, cfg.project, cfg.run_name)
    # this is hard-coded!
    if not this_experiment_folder.exists():
        this_experiment_folder.mkdir(parents=True)

    # 3. Copy config_path to experiment folder
    dst_path = Path(this_experiment_folder, launcher_config_path.name)
    if dst_path.exists():
        warn("Config already exists in experiment folder, overriding it.")
    task_config_path = shutil.copytree(
        launcher_config_path,
        dst_path,
        dirs_exist_ok=True,
    )

    # 4. Hardcode Hydra output dir so that a single one is used (including across processes)
    new_config_name_path = Path(task_config_path, launcher_config_name).with_suffix(
        ".yaml"
    )
    with open(new_config_name_path, "r") as f:
        cfg_file_data = yaml.safe_load(f)
    cfg_file_data["hydra"]["run"]["dir"] = hydra_cfg["run"]["dir"]
    with open(new_config_name_path, "w") as f:
        yaml.dump(cfg_file_data, f)

    return task_config_path, launcher_config_name
