import random
import numpy as np
import torch


class StatsRecorder:
    """Accumulates normalization statistics across mini-batches.
    ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html

    Args:
        red_dims (tuple): Dimensions to reduce over when computing statistics.
    """
    def __init__(self, red_dims=(0,1)):
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """ Updates the running statistics with a new mini-batch.

        Args:
            data (np.ndarray or torch.Tensor): New mini-batch of data of shape (nobservations, ndimensions).
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims, keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]
        else:
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean + 1e-5)**2
            self.std  = torch.sqrt(self.std)
            self.nobservations += n


def create_validation_sampler(dataset_size, validation_split, shuffle_dataset=True, distributed=False):
    """Creates a sampler for train and validation sets.

    Args:
        dataset_size (int): The size of the dataset.
        validation_split (float): The proportion of the dataset to use for validation.
        shuffle_dataset (bool): Whether to shuffle the dataset before splitting.

    Returns:
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the train set.
        torch.utils.data.sampler.SubsetRandomSampler: A sampler for the validation set.
    """
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    if distributed:
        train_sampler = DistributedProxySampler(torch.utils.data.sampler.SubsetRandomSampler(train_indices))
        valid_sampler = DistributedProxySampler(torch.utils.data.sampler.SubsetRandomSampler(val_indices))
    else:
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def libri_collate_fn(batch):
    """Collects together sequences into a single batch, arranged in descending length order.

    Args:
        batch (list(tuple(torch.Tensor, int))): A list of (sequence, label) pairs.

    Returns:
        torch.Tensor: The padded sequences of shape (B, C, L).
        torch.Tensor: The labels.
    """
    batch_size = len(batch)
    batch.sort(key=(lambda x: x[0].shape[-1]), reverse=True)
    x_seq, y_seq, lengths = [], [], torch.zeros(batch_size, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        lengths[i] = len(x_seq)
        x_seq.append(x.T)
        y_seq.append(y)
    padded_x_seq = torch.nn.utils.rnn.pad_sequence(x_seq, batch_first=True)  # (B, L, C)
    if not isinstance(batch[0][1], torch.Tensor):
        padded_y_seq = torch.LongTensor(y_seq)
    else:
        # padded_y_seq = torch.nn.utils.rnn.pad_sequence(y_seq, batch_first=True, padding_value=-1)
        padded_y_seq = torch.stack(y_seq)
    return padded_x_seq.transpose(1,2).contiguous(), padded_y_seq


def seed_worker(worker_id):
    """Seed the worker for reproducibility.

    Args:
        worker_id (int): The worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
# Credits: https://github.com/pytorch/pytorch/issues/23430
class DistributedProxySampler(torch.utils.data.distributed.DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    Note:
        Input sampler is assumed to be of constant size.

    Args:
        sampler (torch.utils.data.Sampler): Input data sampler.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):        
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)
