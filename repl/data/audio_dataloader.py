from repl.data.datasets import LibriDataset
from repl.data.utils import libri_collate_fn, seed_worker
import torch

from torch.utils.data.distributed import DistributedSampler


class AudioDataLoader:
    """Generic audio data loader class.
    
    Args:
        dataset: Dataset class to use.
        data_dir: Directory where data is stored.
        num_workers: Number of workers for data loading.
        collate_fn: Collate function for data loading.
        distributed: Whether to use distributed sampling.
        val_size: Proportion of training data to use for validation.
        **kwargs: Additional arguments for the dataset class.
    """
    def __init__(
        self,
        dataset,
        data_dir,
        num_workers=16,
        collate_fn=None,
        distributed=False,
        val_size=0.1,
        **kwargs
    ):
        self.dataset = dataset
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.sampler = DistributedSampler if distributed else None
        self.val_size = val_size
        self._dataset_kwargs = kwargs

        self._train_subset = "train"
        self._val_subset = "train"
        self._test_subset = "test"
    
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
                sampler = self.sampler(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
        ), sampler

    def get_train(self, batch_size, sampler=None):
        """Get training data loader.
        
        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the training split and the sampler used.
        """
        return self.__get_loader(self._train_subset, batch_size, shuffle=True, sampler=sampler)

    def get_validation(self, batch_size, sampler=None):
        """Get validation data loader.

        Args:
            batch_size: Batch size for the data loader.
            sampler: Sampler to use for the data loader.

        Returns:
            DataLoader for the validation split and the sampler used.
        """
        return self.__get_loader(self._val_subset, batch_size, shuffle=False, sampler=sampler)

    def get_test(self, batch_size):
        """Get test data loader.

        Args:
            batch_size: Batch size for the data loader.

        Returns:
            DataLoader for the test split and the sampler used.
        """
        return self.__get_loader(self._test_subset, batch_size, shuffle=False)


class LibriDataLoader(AudioDataLoader):
    """Data loader for the LibriSpeech dataset.
    
    Args:
        data_dir: Directory where data is stored.
        num_workers: Number of workers for data loading.
        distributed: Whether to use distributed sampling.
        **kwargs: Additional arguments for the dataset class.
            - seq_len: Length of audio sequences.
            - root: Root directory of the dataset.
            - label: Type of labels to use (speaker or phone).
        """
    def __init__(self, data_dir, num_workers=16, distributed=False, **kwargs):
        super().__init__(
            LibriDataset,
            data_dir,
            num_workers,
            collate_fn=libri_collate_fn,
            distributed=distributed,
            **kwargs,
        )

        self._train_subset = "train"
        self._val_subset = "test"
        self._test_subset = "test"

        self.datasets = super()._make_datasets([self._train_subset, self._test_subset])
        # LibriDataset does not have a validation set, use test set for validation
        print("Warning: LibriDataset does not have a validation set, using test set for validation")
