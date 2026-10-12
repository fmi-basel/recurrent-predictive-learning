# credits: https://github.com/loeweX/Greedy_InfoMax

import os

from repl.data.datasets import MouseDataset, MNISTSequencesDataset, PFCOddballImgSequencesDataset
from repl.data.tlt12_wds import LabeledTraversalWebDataset, TraversalWebDataset, TraversalWebDatasetSubset
from repl.data.synthetic_sprites_dataset import SpriteVideoDataset
from repl.data.utils import create_validation_sampler
import torch
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler


class ImageDataLoader:
    """Generic Image DataLoader class for vision datasets.
    
    Args:
        dataset: Dataset class to use.
        data_dir: Directory where the data is stored.
        grayscale: Whether to convert images to grayscale.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply.
        distributed: Whether to use distributed sampling.
        **kwargs: Additional arguments for the dataset class.
    """
    def __init__(
        self,
        dataset,
        data_dir,
        grayscale=False,
        num_workers=16,
        augmentations={},
        distributed=False,
        **kwargs
    ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.grayscale = grayscale
        self.num_workers = num_workers
        self.augmentations = augmentations
        self.sampler = DistributedSampler if distributed else None

        self._train_subset = "train+unlabeled"
        self._train_supervised_subset = "train"
        self._val_supervised_subset = "val"
        self._val_subset = "val"
        self._test_subset = "test"
        self._dataset_kwargs = kwargs

    def get_transforms(self, eval=False):
        """Get the image transformations based on augmentations.
        
        Args:
            eval: Whether the transforms are for evaluation (True) or training (False).
        
        Returns:
            transforms_list: Composed list of transformations.
        """
        transforms_list = []

        # Random cropping for training
        if self.augmentations.get("randcrop", None) and not eval:
            transforms_list.append(
                transforms.RandomCrop(self.augmentations["randcrop"])
            )

        # Center cropping for evaluation
        if self.augmentations.get("randcrop", None) and eval:
            transforms_list.append(
                transforms.CenterCrop(self.augmentations["randcrop"])
            )

        # Grayscale conversion (can also be controlled by dataset)
        # and normalization if specified
        if self.augmentations.get("grayscale", None):
            transforms_list.append(transforms.Grayscale())
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(
                transforms.Normalize(
                    mean=self.augmentations["bw_mean"], std=self.augmentations["bw_std"]
                )
            )
        elif self.augmentations.get("mean", None) and self.augmentations.get("std", None):
            transforms_list.append(transforms.ToTensor())
            transforms_list.append(
                transforms.Normalize(
                    mean=self.augmentations["mean"], std=self.augmentations["std"]
                )
            )
        else:
            transforms_list.append(transforms.ToTensor())

        transforms_list = transforms.Compose(transforms_list)
        return transforms_list

    def _make_datasets(self, splits):
        """Create datasets for the given splits.
        
        Args:
            splits: List of dataset splits to create.
        
        Returns:
            Dictionary of datasets for each split.
        """
        datasets = {}
        for split in splits:
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval = not split.startswith("train")),
                **self._dataset_kwargs,
            )
        return datasets

    def __get_loader(self, split, batch_size, shuffle, sampler=None):
        """Create a data loader for the given split.
        
        Args:
            split: Dataset split to create loader for.
            batch_size: Batch size for the data loader.
            shuffle: Whether to shuffle the data.
            sampler: Sampler to use for the data loader.
            
        Returns:
            DataLoader for the given split and the sampler used.
        """
        dataset = self.datasets[split]
        if self.sampler is not None:
            if sampler is None or (sampler is not None and not isinstance(sampler, DistributedSampler)):
                print("Warning: sampler overriden by distributed sampler")
                sampler = self.sampler(self.dataset)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,
        ), sampler

    def get_train(self, batch_size, sampler=None):
        """Get dataloader for training data.
        
        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.
            
        Returns:
            DataLoader for the training split and the sampler used.
        """
        return self.__get_loader(self._train_subset, batch_size=batch_size, shuffle=True, sampler=sampler)
    
    def get_validation(self, batch_size, sampler=None):
        """Get dataloader for validation data.
        
        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.
            
        Returns:
            DataLoader for the validation split and the sampler used.
        """
        return self.__get_loader(self._val_subset, batch_size=batch_size, shuffle=False, sampler=sampler)
    
    def get_supervised_train(self, batch_size, sampler=None):
        """Get dataloader for supervised training data. (not used)
        
        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.
            
        Returns:
            DataLoader for the supervised training split and the sampler used.
        """
        return self.__get_loader(self._train_supervised_subset, batch_size=batch_size, shuffle=True, sampler=sampler)
    
    def get_supervised_validation(self, batch_size, sampler=None):
        """Get dataloader for supervised validation data. (not used)
        
        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.
        
        Returns:
            DataLoader for the supervised validation split and the sampler used.
        """
        return self.__get_loader(self._val_supervised_subset, batch_size=batch_size, shuffle=False, sampler=sampler)
    
    def get_test(self, batch_size):
        """Get dataloader for test data.
        
        Args:
            batch_size: Batch size for the data loader.

        Returns:
            DataLoader for the test split and the sampler used.
        """
        return self.__get_loader(self._test_subset, batch_size=batch_size, shuffle=False)


class ImageSequencesDataLoader(ImageDataLoader):
    """DataLoader for image sequence datasets like MNIST sequences and oddball PFC images.
    
    Args:
        data_dir: Directory where the data is stored.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply.
        distributed: Whether to use distributed sampling.
        val_size: Proportion of training data to use for validation.
        seq_type: Type of sequence to generate (e.g., triplets).
        seq_len: Length of each image sequence.
        num_sequences: Number of sequences to generate.
        imgs: Type of images to use (mnist or oddballs).
        inter_trial_interval: Number of blank frames between sequences.
    """
    def __init__(self, 
                 data_dir, 
                 num_workers=16, 
                 augmentations={}, 
                 distributed=False, 
                 val_size=0.1,
                 seq_type="triplets",
                 seq_len=20,
                 num_sequences=10000,
                 imgs="mnist",
                 inter_trial_interval=0,
                 ):
        
        if imgs == "mnist":
            dataset = MNISTSequencesDataset
        elif imgs == "oddballs":
            dataset = PFCOddballImgSequencesDataset

        super(ImageSequencesDataLoader, self).__init__(
            dataset=dataset,
            data_dir=data_dir,
            num_workers=num_workers,
            augmentations=augmentations,
            distributed=distributed,
            seq_type=seq_type,
            seq_len=seq_len,
            num_sequences=num_sequences,
            inter_trial_interval=inter_trial_interval,
        )

        if distributed:
            raise NotImplementedError("Distributed training not supported yet for MNIST")
            self.sampler = DistributedSampler
        else:
            self.sampler = None

        self._train_subset = "train"
        self._val_subset = "train"
        self._test_subset = "test"

        self.datasets = super()._make_datasets([self._train_subset, self._test_subset])
        self.train_sampler, self.val_sampler = create_validation_sampler(len(self.datasets[self._train_subset]), val_size, distributed)
        
    def get_train(self, batch_size):
        """Get dataloader for training data."""
        return super().get_train(batch_size, self.train_sampler)
    
    def get_validation(self, batch_size):
        """Get dataloader for validation data."""
        return super().get_validation(batch_size, self.val_sampler)

class SpriteVideoDataLoader(ImageDataLoader):
    """
    DataLoader for the SpriteVideoDataLoader dataset that generates synthetic videos
    of moving sprites with various trajectories and transformations.
    
    Args:
        data_dir: Directory where the data is stored.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply.
        distributed: Whether to use distributed sampling.
        val_size: Proportion of training data to use for validation.
        output_size: Size of output frames (width, height).
        seq_len: Length of each video sequence.
        num_sequences: Number of sequences to generate.
        seed: Random seed for reproducibility.
        background: Background color (grayscale level) for the video frames.
        max_sprites: Maximum number of sprites (i.e., image categories).
        exclude_latent_regions: Whether to exclude some latent regions from the sampling space during training.
        discretize_latents: Whether to use discretized latent variables to generate the videos.
        noise_type: Type of noise to add to frames (gaussian or salt_pepper). Default: None.
        noise_level: Level of noise to add to frames (float between 0 and 1).
        frozen_noise: Whether to use fixed noise patterns (across frames in each sequence).
        noise_on_top: Whether to add noise on top of the frame or only to the background.
        grid_enabled: Whether to use a grid overlay.
        frozen_grid: Whether to use a fixed grid pattern (across frames in each sequence).
        sprite_imgs: Name of the sprite images to use (should be a folder name with images in data_dir).
        grayscale: Whether to convert frames to grayscale.
        occlude_n_frames: Number of frames to occlude (e.g., frame-dropout) in each sequence.
    """
    
    def __init__(self, 
                 data_dir, 
                 num_workers=16, 
                 augmentations={},
                 distributed=False, 
                 val_size=0.1,
                 output_size=(64, 64),
                 seq_len=32,
                 num_sequences=100000,
                 seed=42,
                 background=0.5,
                 max_sprites=16,
                 exclude_latent_regions=False,
                 discretize_latents=False,
                 noise_type=None,
                 noise_level=0.1,
                 frozen_noise=False,
                 noise_on_top=False,
                 grid_enabled=False,
                 frozen_grid=False,
                 sprite_imgs="animals",
                 grayscale=False,
                 occlude_n_frames=0,
                 ):
        # Initialize with grayscale=False since SpriteVideoDataset generates RGB videos
        super(SpriteVideoDataLoader, self).__init__(
            dataset=SpriteVideoDataset,
            data_dir=data_dir,
            grayscale=False,
            num_workers=num_workers,
            augmentations=augmentations,
            distributed=distributed,
            # Parameters specific to SpriteVideoDataset
            output_size=output_size,
            seq_len=seq_len,
            num_sequences=num_sequences,
            seed=seed,
            background=background,
            sprite_img_dir=sprite_imgs,
            discretize_latents=discretize_latents,
            max_sprites=max_sprites,
            noise_type=noise_type,
            noise_intensity=noise_level,
            freeze_noise=frozen_noise,
            noise_on_top=noise_on_top,
            grid_enabled=grid_enabled,
            freeze_grid=frozen_grid,
            occlude_n_frames=occlude_n_frames,
        )

        if distributed:
            raise NotImplementedError("Distributed training not supported yet for SpriteVideo")
            self.sampler = DistributedSampler
        else:
            self.sampler = None

        # Define subsets for training and testing
        self._train_subset = "train"
        self._val_subset = "train"
        self._test_subset = "test"
        
        self._train_supervised_subset = "train"
        self._val_supervised_subset = "train"

        self.exclude_latent_regions = exclude_latent_regions

        self.grayscale = grayscale

        # Create datasets
        self.datasets = self._make_datasets([self._train_subset, self._test_subset])
        
        # Create train and validation samplers from the training dataset
        self.train_sampler, self.val_sampler = create_validation_sampler(
            len(self.datasets[self._train_subset]), 
            val_size, 
            distributed
        )

    def _make_datasets(self, splits):
        """Create datasets for the given splits.
        
        Args:
            splits: List of dataset splits to create.

        Returns:
            Dictionary of datasets for each split.
        """
        datasets = {}
        if "train" in splits:
            datasets["train"] = self.dataset(
                self.data_dir,
                split="train",
                download=True,
                transform=self.get_transforms(eval = False),
                exclude_latent_regions=self.exclude_latent_regions,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
            mean, std = datasets["train"].mean, datasets["train"].std
        for split in splits:
            if split == "train":
                continue
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=True,
                transform=self.get_transforms(eval = not split.startswith("train")),
                mean=mean,
                std=std,
                exclude_latent_regions=False,
                grayscale=self.grayscale,
                **self._dataset_kwargs,
            )
        return datasets
        
    def get_train(self, batch_size):
        """Get dataloader for training data"""
        return super().get_train(batch_size, self.train_sampler)
    
    def get_validation(self, batch_size):
        """Get dataloader for validation data"""
        return super().get_validation(batch_size, self.val_sampler)
    

class MouseVideoDataLoader(ImageDataLoader):
    """
    DataLoader for the MouseVideo dataset.
    
    Args:
        data_dir: Directory where the data is stored.
        num_workers: Number of workers for data loading.
        augmentations: Dictionary of augmentations to apply. (not used)
        distributed: Whether to use distributed sampling.
        val_size: Proportion of training data to use for validation. (not used)
        seq_len: Length of each video sequence.
        num_sequences: Number of sequences to generate. (not used)
        grayscale: Whether to convert frames to grayscale.
    """
    def __init__(self, 
                 data_dir, 
                 num_workers=16, 
                 augmentations={},
                 distributed=False, 
                 val_size=0.1,
                 seq_len=25,
                 num_sequences=100000,
                 grayscale=True,
                 ):
        """
        DataLoader for the MouseVideo dataset.
        
        Args:
            data_dir (str): The directory containing mouse video data
            num_workers (int): Number of worker processes for data loading
            augmentations (dict): Dictionary of augmentations to apply
            distributed (bool): Whether to use distributed training
            val_size (float): Proportion of training data to use for validation
            seq_len (int): Length of each video sequence
            num_sequences (int): Number of sequences to generate
        """
        
        # Initialize with grayscale=True since MouseVideoDataset generates grayscale videos
        super(MouseVideoDataLoader, self).__init__(
            dataset=MouseDataset,
            data_dir=data_dir,
            grayscale=True,
            num_workers=num_workers,
            distributed=distributed,
            seq_len=seq_len,
        )
        if distributed:
            raise NotImplementedError("Distributed training not supported yet for MouseVideo")
            self.sampler = DistributedSampler
        else:
            self.sampler = None

        # Define subsets for training and testing
        self._train_subset = "train"
        self._val_subset = "val"
        self._test_subset = "test"

        self._train_supervised_subset = "train"
        self._val_supervised_subset = "val"
        self.grayscale = grayscale

        # Create datasets
        self.datasets = self._make_datasets([self._train_subset, self._val_subset, self._test_subset])

    def _make_datasets(self, splits):
        """Create datasets for the given splits.
        
        Args:
            splits: List of dataset splits to create.
            
        Returns:
            Dictionary of datasets for each split.
        """
        datasets = {}
        if "train" in splits:
            datasets["train"] = self.dataset(
                self.data_dir,
                split="train",
                download=False,
                grayscale=self.grayscale,
                velocity_stats=0.0,  # Placeholder, will be set from training dataset, velocities will be normalized. Pass None to avoid normalization.
                **self._dataset_kwargs,
            )
            velocity_stats = datasets["train"].velocity_stats
        for split in splits:
            if split == "train":
                continue
            datasets[split] = self.dataset(
                self.data_dir,
                split=split,
                download=False,
                grayscale=self.grayscale,
                velocity_stats=velocity_stats,
                **self._dataset_kwargs,
            )
        return datasets


class TLT12DataLoader(ImageDataLoader):
    def __init__(
        self,
        data_dir,
        num_workers=16,
        augmentations={},
        distributed=False,
        seed=42,
        normalize_latents=True,
    ):
        data_dir = os.path.join(data_dir, "TLT12", "shards")
        super(TLT12DataLoader, self).__init__(
            dataset=TraversalWebDataset,
            data_dir=data_dir,
            num_workers=num_workers,
            distributed=distributed,
            seed=seed
        )

        if distributed:
            raise NotImplementedError("Distributed training not yet supported for TLT12")

        self._train_subset = "train"
        self._val_subset = "val"
        self._test_subset = "val"
        self.sampler = None
        self.normalize_latents = normalize_latents

        self._webdataset_handler()

    def _webdataset_handler(self):
        # check if files in data_dir/train/ have a tar.gz extension or a tar extension
        import os

        train_files = os.listdir(f"{self.data_dir}/train")
        if any(f.endswith(".tar.gz") for f in train_files):
            shard_pattern = "shard-*.tar.gz"
        else:
            shard_pattern = "shard-*.tar"

        self.train_traversal_dataset = self.dataset(
            urls=f"{self.data_dir}/train/{shard_pattern}",
            info_dir=f"{self.data_dir}/train",
            transform=transforms.Compose([transforms.Resize((64, 64)),])
        )
        self.val_traversal_dataset = self.dataset(
            urls=f"{self.data_dir}/val/{shard_pattern}",
            info_dir=f"{self.data_dir}/val",
            transform=transforms.Compose([transforms.Resize((64, 64)),]),
            shuffle=0
        )
        self.test_traversal_dataset = self.dataset(
            urls=f"{self.data_dir}/test/{shard_pattern}",
            info_dir=f"{self.data_dir}/test",
            transform=transforms.Compose([transforms.Resize((64, 64)),]),
            shuffle=0
        )
        self.train_subset = TraversalWebDatasetSubset(
            self.train_traversal_dataset, n_shards=5
        )
        self.ood_test_dataset = self.dataset(
            urls=f"{self.data_dir}/ood-test/{shard_pattern}",
            info_dir=f"{self.data_dir}/ood-test",
            transform=transforms.Compose([transforms.Resize((64, 64)),]),
            shuffle=0
        )

        self.get_train = lambda batch_size, sampler=None: (
            LabeledTraversalWebDataset(self.train_traversal_dataset, normalize_latents=self.normalize_latents).loader(
                batch_size=batch_size,
                num_workers=self.num_workers,
            ), sampler)
        self.get_validation = lambda batch_size, sampler=None: (
            LabeledTraversalWebDataset(self.val_traversal_dataset, normalize_latents=self.normalize_latents).loader(
                batch_size=batch_size,
                num_workers=2,
            ), sampler)
        self.get_test = lambda batch_size, sampler=None: (
            LabeledTraversalWebDataset(self.test_traversal_dataset, normalize_latents=self.normalize_latents).loader(
                batch_size=batch_size,
                num_workers=2,
            ), sampler)
        self.get_supervised_train = lambda batch_size, sampler=None: (
            LabeledTraversalWebDataset(self.train_subset, normalize_latents=self.normalize_latents).loader(
                batch_size=batch_size,
                num_workers=2,
            ), sampler)
        self.get_supervised_validation = self.get_validation

    def get_ood_test(self, batch_size, sampler=None):
        """Get dataloader for out-of-distribution test data."""
        return LabeledTraversalWebDataset(self.ood_test_dataset, normalize_latents=self.normalize_latents).loader(
            batch_size=batch_size,
            num_workers=2,
        ), sampler
        