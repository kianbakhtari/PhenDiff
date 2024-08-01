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

import random
from argparse import Namespace
from pathlib import Path
from typing import Optional
import os
import glob
from PIL import Image
from io import BytesIO
from typing import Literal, Optional, Tuple, Union, List, Dict, Any

import torch
from accelerate.logging import MultiProcessAdapter
from datasets import load_dataset
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from torchvision.io import read_image


class NoLabelsDataset(ImageFolder):
    """A custom dataset that only returns the images, without their labels."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            sample
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class JointTransforms:
    """A custom image transformation and data augmentation class for paired samples."""
    def __init__(self, args, h_flip_prob=0.5, v_flip_prob=0.5):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

        if isinstance(args.definition, int):
            (h, w) = (args.definition, args.definition)
        else:
            (h, w) = args.definition

        self.default_transforms = transforms.Compose([
            transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # map to [-1, 1] for SiLU
        ])

    def __call__(self, imgA, imgB):

        imgA = self.default_transforms(imgA)
        imgB = self.default_transforms(imgB)

        if random.random() < self.h_flip_prob:
            imgA = F.hflip(imgA)
            imgB = F.hflip(imgB)

        if random.random() < self.v_flip_prob:
            imgA = F.vflip(imgA)
            imgB = F.vflip(imgB)

        return imgA, imgB
    

class PairedSamplesDataset(Dataset):
    """A custom dataset that outputs paired samples.
    Data should be organized in this way:
        /root
            /class_A
                img1.png
                img2.png
                ...
            /class_B
                img1.png
                img2.png
        ...
    root: root directory for the paired dataset (should have the classes under itself like the illustration above)
    source_class: for one-way translation tasks involving source and target classes, like segmentation.
    """

    def __init__(self, args, root : str, source_class : Optional[str] = None, transform: JointTransforms = None):
        root = os.path.abspath(root)
        class_directories = glob.glob(os.path.join(root, '*'))
        assert len(class_directories) == 2, f"There sould be two and only two classes under the root: {root}"

        if not all([os.path.isdir(d) for d in class_directories]):
            raise Exception(f"There should be only directories under the root: {root}")
        
        classA_dir, classB_dir = class_directories
        assert classA_dir != classB_dir, f"Two classes should be different: {classA_dir}, {classB_dir}"
        assert sorted(os.listdir(classA_dir)) == sorted(os.listdir(classB_dir)), "Paired data should be organazied in the way described above,\
        with samples having identical names under two classes."

        self.classes, self.class_to_idx = self.find_classes(root)

        self.classA_dir = os.path.join(root, self.classes[0]) # A -> 0
        self.classB_dir = os.path.join(root, self.classes[1]) # B -> 1
        self.images = sorted(os.listdir(self.classA_dir))
        self.transform = transform
        self.source_class = source_class
        if self.source_class is not None:
            assert self.source_class in self.classes, f"source_class {source_class} incompatible with classes: {self.classes}"
        
        self.convert_to_rgb = True
        if args.denoiser_in_channels == 1:
            self.convert_to_rgb = False

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.
        Borrowed from PyTorch: https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html#ImageFolder
        See :class:`DatasetFolder` for details.
        We use this function to find the class indicies because the ImageFolder (which is handling the unpaired dataset) is using this function internally.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imgA_path = os.path.join(self.classA_dir, self.images[idx])
        imgB_path = os.path.join(self.classB_dir, self.images[idx])

        if self.convert_to_rgb:
            imgA = Image.open(imgA_path).convert("RGB")
            imgB = Image.open(imgB_path).convert("RGB")
        else:
            imgA = Image.open(imgA_path)
            imgB = Image.open(imgB_path)
        
        if self.transform is not None:
            imgA, imgB = self.transform(imgA, imgB)
        
        # Always source class comes first
        if self.source_class is not None:
            if self.classes[1] == self.source_class:
                return {
                    "source_images": imgB,
                    "source_class": self.class_to_idx[self.classes[1]],
                    "target_images": imgA,
                    "target_class": self.class_to_idx[self.classes[0]],
                }
            
        return {
                "source_images": imgA,
                "source_class": self.class_to_idx[self.classes[0]],
                "target_images": imgB,
                "target_class": self.class_to_idx[self.classes[1]],
            }


def setup_dataset(
    args: Namespace, logger: MultiProcessAdapter
) -> tuple[ImageFolder | Subset, NoLabelsDataset | Subset, int]:
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    #################################################
    # def default_loader(path: str) -> Image.Image:
    #     with open(path, "rb") as f:
    #         img = Image.open(BytesIO(f.read()))
    #         # img_array = np.array(img)
    #         # print("\n\n\nPIl LOADER")
    #         # print(img_array.shape)
    #         return img

    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(BytesIO(f.read()))
            return img

    def accimage_loader(path: str) -> Any:
        import accimage

        try:
            return accimage.Image(path)
        except OSError:
            # Potentially a decoding problem, fall back to PIL.Image
            return pil_loader(path)

    def default_loader(path: str) -> Any:
        from torchvision import get_image_backend

        if get_image_backend() == "accimage":
            return accimage_loader(path)
        else:
            return pil_loader(path)
    #################################################

    if args.dataset_name is not None:
        raise NotImplementedError("Not tested yet")
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            split="train",
        )
    elif args.use_pytorch_loader:
        if args.denoiser_in_channels == 1:
            dataset: ImageFolder | Subset = ImageFolder(
                root=Path(args.train_data_dir, args.split).as_posix(),
                transform=lambda x: transformations(x),
                target_transform=lambda y: torch.tensor(y).long(),
                loader=default_loader,
            )
            raw_dataset: NoLabelsDataset | Subset = NoLabelsDataset(
                root=Path(args.train_data_dir, args.split).as_posix(),
                transform=lambda x: raw_transformations(x),
                loader=default_loader,
            )
        else:
            dataset: ImageFolder | Subset = ImageFolder(
                root=Path(args.train_data_dir, args.split).as_posix(),
                transform=lambda x: transformations(x),
                target_transform=lambda y: torch.tensor(y).long(),
            )
            raw_dataset: NoLabelsDataset | Subset = NoLabelsDataset(
                root=Path(args.train_data_dir, args.split).as_posix(),
                transform=lambda x: raw_transformations(x),
            )
        assert len(dataset) == len(
            raw_dataset
        ), "dataset and raw_dataset should have the same length"
    else:
        raise NotImplementedError("Not tested yet")
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    if args.perc_samples is not None:
        logger.warning(
            f"Subsampling the training dataset to {args.perc_samples}% of samples per class"
        )
        if not args.compute_metrics_full_dataset:
            logger.warning(
                f"Metrics computation will be done against the subsampled dataset"
            )
            dataset, raw_dataset = _select_subset_of_dataset(args, dataset, raw_dataset)
        else:
            dataset = _select_subset_of_dataset(args, dataset)  # type: ignore

    # Preprocessing the datasets and DataLoaders creation
    # transforms for the training dataset
    if isinstance(args.definition, int):
        (h, w) = (args.definition, args.definition)
    else:
        (h, w) = args.definition
    list_transforms = [
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # map to [-1, 1] for SiLU
    ]
    if args.data_aug_on_the_fly: # Kian: ture by default.
        # Kian: There is a "no_data_aug_on_the_fly" arg in the arguments. We should put it in the args to disable augmentations.
        list_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    transformations = transforms.Compose(list_transforms)
    # transforms for the raw dataset
    raw_transformations = transforms.Compose(
        [
            transforms.Resize(
                (h, w), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.PILToTensor(),
        ]
    )

    def transform_images(examples):
        images = [transformations(image.convert("RGB")) for image in examples["image"]]
        class_labels = examples["label"]
        return {"images": images, "class_labels": class_labels}

    if not args.use_pytorch_loader:
        raise NotImplementedError("Not tested yet")
        dataset.set_transform(transform_images)

    return dataset, raw_dataset, len(dataset.classes)


def setup_paired_dataset(
    args: Namespace, logger: MultiProcessAdapter, test_split:bool=False
) -> PairedSamplesDataset:
    
    assert args.use_pytorch_loader, "You should use args.use_pytorch_loader for using paired samples"

    if test_split:
        test_dataset: PairedSamplesDataset = PairedSamplesDataset(
        args,
        root=Path(args.test_data_dir).as_posix(), 
        source_class=args.source_class_for_paired_training,
        transform=JointTransforms(args, h_flip_prob=0, v_flip_prob=0),
        )
        return test_dataset

    paired_dataset: PairedSamplesDataset = PairedSamplesDataset(
        args,
        root=Path(args.paired_train_data_dir, args.split).as_posix(), # Kian: args.split is "train" by default.
        source_class=args.source_class_for_paired_training,
        transform=JointTransforms(args),
    )
    return paired_dataset


def _select_subset_of_dataset(
    args: Namespace,
    full_dataset: ImageFolder,
    full_raw_dataset: Optional[NoLabelsDataset] = None,
) -> tuple[Subset, Subset] | Subset:
    """Subsamples the given dataset(s) to have <perc_samples>% of each class."""

    # 1. First test if the dataset is balanced; for now we assume it is
    class_counts = dict.fromkeys(
        [full_dataset.class_to_idx[cl] for cl in full_dataset.classes], 0
    )
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    nb_classes = len(class_counts)

    # print()
    # print(f"list(class_counts.values()) --> {list(class_counts.values())}")
    # print(f"[class_counts[0]] * nb_classes --> {[class_counts[0]] * nb_classes}")
    # print()

    assert (
        list(class_counts.values()) == [class_counts[0]] * nb_classes
    ), "The dataset is not balanced between classes"

    # 2. Then manually sample <perc_samples>% of each class
    orig_nb_samples_per_balanced_classes = class_counts[0]

    nb_selected_samples_per_class = int(
        orig_nb_samples_per_balanced_classes * args.perc_samples / 100
    )

    sample_indices = []

    nb_selected_samples = dict.fromkeys(
        [full_dataset.class_to_idx[cl] for cl in full_dataset.classes], 0
    )

    # set seed
    # `random` is only used here, for the dataset subsampling
    random.seed(args.seed)

    # random.sample(x, len(x)) shuffles x out-of-place
    iterable = random.sample(list(enumerate(full_dataset.samples)), len(full_dataset))

    for idx, (_, class_label) in iterable:
        # stop condition
        if (
            list(nb_selected_samples.values())
            == [nb_selected_samples_per_class] * nb_classes
        ):
            break
        # select sample
        if nb_selected_samples[class_label] < nb_selected_samples_per_class:
            sample_indices.append(idx)
            nb_selected_samples[class_label] += 1

    assert (
        len(sample_indices) == nb_selected_samples_per_class * nb_classes
    ), "Something went wrong in the subsampling..."

    # 3. Return the subset(s)
    subset = Subset(full_dataset, sample_indices)
    if full_raw_dataset is not None:
        raw_subset = Subset(full_raw_dataset, sample_indices)
        assert subset.indices == raw_subset.indices
    else:
        raw_subset = None

    # hacky but ok to do this because each class is present in the subset
    subset.classes = full_dataset.classes
    if full_raw_dataset is not None:
        raw_subset.classes = full_raw_dataset.classes

    subset.targets = [full_dataset.targets[i] for i in subset.indices]
    if full_raw_dataset is not None:
        raw_subset.targets = [full_raw_dataset.targets[i] for i in raw_subset.indices]

    if full_raw_dataset is not None:
        return subset, raw_subset
    else:
        return subset
    

# TODO
def default_loader(path: str):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img
    



"""

Img2Img:
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
        



training.py:
<class 'src.pipeline_conditional_ddim.pipeline_conditionial_ddim.ConditionalDDIMPipeline'>
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


perform_class_transfer_for_training:
<class 'src.pipeline_conditional_ddim.pipeline_conditionial_ddim.ConditionalDDIMPipeline'>
        ConditionalDDIMPipeline {
        "_class_name": "ConditionalDDIMPipeline",
        "_diffusers_version": "0.18.2",
        "scheduler": [
            "diffusers",
            "DDIMScheduler"
        ],
        "unet": [
            "torch",
            "DistributedDataParallel"
        ]
        }


"""
