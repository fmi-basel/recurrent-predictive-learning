import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

import random

from repl.data.dataset_assets import create_dict_from_phones, download_librispeech_with_phonemes, libri_flist_reader

from torchvision.datasets import MNIST


class LibriDataset:
    """ LibriSpeech dataset (Panayotov, Vassil, et al. ICASSP (2015))
    Parts of the code adapted from:
        https://github.com/loeweX/Greedy_InfoMax/blob/master/GreedyInfoMax
    to be in compliance with Oord et al., 2018 CPC paper.
    
    Args:
        data_dir (str): The directory to store the dataset
        root (str): The root directory of the dataset
        split (str): The dataset split to use, can be 'train' or 'test'
        download (bool): Whether to download the dataset
        seq_len (int): The length of the audio sequences
        label (str): Which labels to use speaker or phone
        cache_fname (str): The filename to cache the dataset (not implemented yet)

    Attributes:
        valid_splits (list): The valid dataset splits
        valid_labels (list): The valid labels
        split (str): The dataset split to use
        data_dir (str): The directory to store the dataset
        root (str): The root directory of the dataset
        cache_fname (str): The filename to cache the dataset
        length (int): The length of the audio sequences
        label (str): Which labels to use
        file_list (list): The list of files in the dataset
        speaker_dict (dict): A dictionary mapping speaker ids to file indices
        speaker_dict_map (dict): A dictionary mapping speaker ids to integers
        phone_dict (dict): A dictionary mapping filenames to phone labels
        num_classes (int): The number of classes
        mean (float): The mean of the training dataset
        std (float): The standard deviation of the training dataset
    """
    valid_splits = ["train", "test"]
    valid_labels = ["speaker", "phone"]
    def __init__(
            self,
            data_dir,
            root,
            split="train",
            download=False,
            seq_len=20480,
            label="speaker",
            cache_fname=None,
    ):
        # check validity of split and label
        assert split in self.valid_splits, "split must be one of {}".format(self.valid_splits)
        assert label in self.valid_labels, "label must be one of {}".format(self.valid_labels)

        if download:
            download_librispeech_with_phonemes(data_dir)

        self.split = split
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, root)
        self.cache_fname = cache_fname
        self.length = seq_len

        flist = os.path.join(self.data_dir, "LibriSpeech100_labels_split", f"{split}_split.txt")
        assert os.path.exists(flist), "flist {} not found.".format(flist)

        self.file_list, self.speaker_dict = libri_flist_reader(flist)

        self.label = label
        if self.label == "speaker":
            self.num_classes = 251
            # map speaker ids to integers
            self.speaker_dict_map = dict(zip(sorted(self.speaker_dict.keys()), range(251)))
        else:
            self.num_classes = 41
            # dictionary mapping filenames to phone labels
            self.phone_dict = self.load_phone_dict() if self.label == "phone" else None

        self.mean = -22.77745257541822
        self.std = 2068.438282538686

    def get_data_chunked(self, audio):
        """Get a random chunk of audio data of length `self.length`.
        
        Args:
            audio (torch.Tensor): The audio data
        Returns:
            torch.Tensor: The audio chunk of length `self.length`
        """
        # discard last part that is not a full 10ms
        max_length = audio.shape[1] // 160 * 160
        start_idx = np.random.choice(
            np.arange(160, max_length - self.length - 0, 160)
        )

        audio = audio[:, start_idx : start_idx + self.length].to(torch.float32)
        return audio
    
    def get_data_full(self, audio):
        """Get the full audio data, truncated to a multiple of 10ms.
        Used for phone classification.
        
        Args:
            audio (torch.Tensor): The audio data
        Returns:
            torch.Tensor: The full audio data
        """
        # discard last part that is not a full 10ms
        max_length = audio.shape[1] // 160 * 160
        audio = audio[:, :max_length].to(torch.float32)
        return audio

    def __getitem__(self, index):
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
        fname = os.path.join(self.root, speaker_id, dir_id, f"{filename}.flac")
        audio, sample_rate = torchaudio.load(fname, normalize=False)

        target = self.speaker_dict_map[speaker_id] if self.label == "speaker" else self.phone_dict[filename]
        
        assert (
            sample_rate == 16000
        ), "Inconsistent sample rate throughout the dataset!"

        audio = self.get_data_chunked(audio) if self.label == "speaker" else self.get_data_full(audio)

        audio = (audio - self.mean) / self.std
        return audio, target
        
    def __len__(self):
        return len(self.file_list)
    
    def load_phone_dict(self):
        """Get the phone dictionary mapping filenames to phone labels.
         Returns:
            dict: The phone dictionary
        """
        filename = os.path.join(self.cache_fname, "phone_dict.pkl") if self.cache_fname is not None else None

        if filename is None or not os.path.exists(filename):
            return create_dict_from_phones(
                os.path.join(
                    self.data_dir,
                    "LibriSpeech100_labels_split/converted_aligned_phones.txt",
                ),
                filename,
            )
        else:
            with open(filename, "rb") as f:
                return pickle.load(f)
        
    def get_audio_by_speaker(self, speaker_id, batch_size):
        """Get a batch of audio data for a given speaker.

        Args:
            speaker_id (str): The ID of the speaker
            batch_size (int): The number of audio samples to retrieve
        Returns:
            torch.Tensor: A batch of audio samples for the given speaker
        """
        batch_size = min(len(self.speaker_dict[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[speaker_id][idx]
            )
        return batch


class ImageSequencesDataset(Dataset):
    """Base class for image sequences datasets
    
    Args:
        root (str): The root directory of the dataset
        split (str): The dataset split to use (train/test)
        download (bool): Whether to download the dataset
        transform (callable, optional): A function/transform to apply to the images
        seq_type (str): The type of sequences to generate, e.g., 'triplets', 'local_global_oddball'
        seq_len (int): The length of the sequences to generate
        num_sequences (int): The number of sequences to generate
        num_classes (int): The number of classes in the dataset
        inter_trial_interval (int): The interval between trials (used in local_global_oddball)

    Attributes:
        imgs_by_class (list): A list of images for each class
        dense_labels (torch.Tensor): The dense labels for the sequences
        sequence_sample_labels (torch.Tensor): The labels used for sampling sequences
        root (str): The root directory of the dataset
        split (str): The dataset split to use
        seq_type (str): The type of sequences to generate
        seqlen (int): The length of the sequences to generate
        num_sequences (int): The number of sequences to generate
        num_classes (int or tuple): The number of classes in the dataset
        transform (callable, optional): A function/transform to apply to the images
        inter_trial_interval (int): The interval between trials
    """
    valid_splits = ["train", "test"]

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        seq_type="triplets",
        seq_len=64,
        num_sequences=10000,
        num_classes=10,
        inter_trial_interval=0,
    ):
        super().__init__()
        self.root = os.fspath(root)
        self.split = split
        self.seq_type = seq_type
        self.seqlen = seq_len
        self.num_sequences = num_sequences
        self.num_classes = num_classes
        self.transform = transform
        self.inter_trial_interval = inter_trial_interval

        self._load_base_dataset(download=download)

        self._generate_sequences()

        self._sample_images_for_sequences()

    def _load_base_dataset(self, download):
        """
        Load the base dataset. Must be implemented by subclasses.
        Should set self.imgs_by_class which is a list where each element contains
        images of a specific class.
        """
        raise NotImplementedError("Subclasses must implement _load_base_dataset")

    def _sample_images_for_sequences(self):
        """Sample images for the generated sequences - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _sample_images_for_sequences")


    def _generate_sequences(self):
        """
        Generate sequences based on the sequence type.
        """

        if self.seq_type == "local_global_oddball":
            self.dense_labels = self._generate_local_global_oddball_sequences()
            print(f"Warning: seqlen is overwritten to {self.seqlen} for local_global_oddball")
            self.labels = None # not used
            self.sequence_sample_labels = self.dense_labels[:, :, 0] # use the digit labels for sampling
        elif self.seq_type == "triplets":            
            self.dense_labels = self._generate_triplet_sequences()
            self.labels = None # not used
            self.sequence_sample_labels = self.dense_labels[:, :, 0]

        if self.labels is not None:
            if len(self.labels.shape) == 1:
                self.labels = self.labels.unsqueeze(-1)
        if self.dense_labels is not None:
            if len(self.dense_labels.shape) == 2:
                self.dense_labels = self.dense_labels.unsqueeze(-1)

    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, n):
        x = self.data[n] # shape: (self.seqlen, 28, 28)
        y = self.labels[n] if self.labels is not None else torch.tensor([]) # shape: (num_tasks)
        dense_y = self.dense_labels[n] if self.dense_labels is not None else torch.tensor([]) # shape: (self.seqlen, num_tasks)
        x = x.unsqueeze(1) # shape: (seq_len, 1, 28, 28)
        return x, (y, dense_y, torch.tensor([]), torch.tensor([]), torch.tensor([])) # cts_y, cts_dense_y, aux_labels are not used in this dataset

    def _generate_triplet_sequences(self, n=None):
        """
        Generate sequences of triplets from the same cluster ({1,2,3}, {4,5,6}, {7,8,9}) 
        with zeros only separating different clusters.
        
        The sequence structure is: 
        - For same cluster: [digit1, digit2, digit3, digit1, digit2, digit3, ...]
        - When changing clusters: [digit1, digit2, digit3, 0, digit1, digit2, digit3, ...]
        
        Each triplet sequence has digit labels, cluster labels, and triplet type labels:
        - Digit labels: the actual digit (0-9)
        - Cluster labels: 0 for digit 0, 1 for {1,2,3}, 2 for {4,5,6}, 3 for {7,8,9}
        - Triplet labels: 0-17 for different triplet patterns, -1 for separator (0)
        """
        if n is None:
            n = self.num_sequences
        
        # Labels: [digit, cluster, triplet_type]
        dense_labels = torch.zeros((n, self.seqlen, 3), dtype=torch.long)
        
        # Define clusters
        clusters = [
            [1, 2, 3],  # Cluster 1
            [4, 5, 6],  # Cluster 2
            [7, 8, 9]   # Cluster 3
        ]
        
        for i in range(n):
            pos = 0  # Current position in the sequence
            current_cluster_idx = np.random.randint(0, 3)  # Start with a random cluster
            
            while pos < self.seqlen:
                # Choose whether to change the cluster (initially, we use the starting cluster)
                if pos > 0 and np.random.rand() < 0.2:
                    # Pick a new cluster that's different from the current one
                    new_cluster_idx = np.random.choice([idx for idx in range(3) if idx != current_cluster_idx])
                    
                    # Insert a zero separator before changing clusters
                    dense_labels[i, pos, 0] = 0  # Digit label
                    dense_labels[i, pos, 1] = 0  # Cluster label
                    dense_labels[i, pos, 2] = -1  # Triplet type label
                    pos += 1
                    
                    current_cluster_idx = new_cluster_idx

                    if pos >= self.seqlen:
                        break
                
                # Get the current cluster
                cluster = clusters[current_cluster_idx]
                
                # Generate a random permutation of the digits in the cluster
                digits_perm = np.random.permutation(cluster)
                
                # Calculate the triplet type (0-17)
                if np.array_equal(digits_perm, [cluster[0], cluster[1], cluster[2]]):
                    perm_type = 0
                elif np.array_equal(digits_perm, [cluster[0], cluster[2], cluster[1]]):
                    perm_type = 1
                elif np.array_equal(digits_perm, [cluster[1], cluster[0], cluster[2]]):
                    perm_type = 2
                elif np.array_equal(digits_perm, [cluster[1], cluster[2], cluster[0]]):
                    perm_type = 3
                elif np.array_equal(digits_perm, [cluster[2], cluster[0], cluster[1]]):
                    perm_type = 4
                elif np.array_equal(digits_perm, [cluster[2], cluster[1], cluster[0]]):
                    perm_type = 5
                else:
                    perm_type = 0
                
                # Final triplet type: cluster_idx * 6 + perm_type
                triplet_type = current_cluster_idx * 6 + perm_type
                
                # Add the three digits to the sequence (or as many as will fit)
                for k in range(3):
                    if pos < self.seqlen:  # Only add the digit if there's space
                        digit = digits_perm[k]
                        
                        # Set labels
                        dense_labels[i, pos, 0] = digit  # Digit label
                        dense_labels[i, pos, 1] = current_cluster_idx + 1  # Cluster label (1-based)
                        dense_labels[i, pos, 2] = triplet_type if k == 1 else -1  # Triplet type label (middle digit only)
                        
                        pos += 1
                        
        return dense_labels
    
    def _generate_local_global_oddball_sequences(self, n=None):
        """Generate local-global oddball sequences.
        Each sequence first samples two different digits from {0,1}, {2,3}, {4,5}, {6,7}, {8,9}.
        Then, it randomly chooses 0000, 1111, 1110, or 0001 as the context and repeats this 50 times.
        For the next 150 repetitions, 80% of the time it repeats the context, and 20% of the time it switches to the global oddball for that context.
        So 0000 becomes 0000 0000 ... 0000 (50 times) 0000 (120 times) 0001 (30 times) ....
        The labels are:
        - Digit labels: the actual digit (0-9)
        - Local labels: 0 for the first digit of the pair, 1 for the second digit of the pair
        - Serial position labels: 0, 1, 2, 3 for the position in the context
        - Global context: 0 for xxxx, 1 for xxxy at the third position
        - Local oddball: -1 for first three elements, 0 for non-oddball, 1 for oddball at the last position
        - Global oddball: -1 for first three elements, 0 for non-oddball, 1 for oddball at the last position (global oddball is defined as not matching the context)

        Fixed sequence length to 2400 depending on inter_trial_interval (overwritten in __init__)
        Every other element is a dummy (blank) element. So effectively 800 elements with
        6 labels in total.
        """
        if n is None:
            n = self.num_sequences

        trial_len = 4          # 4 stimuli per trial
        n_trials = 20         # 5 habituation + 15 test
        # n_trials = 100         # 25 habituation + 75 test
        intra_len = trial_len * 2  # 4 stimuli + 4 dummy = 8 steps
        lag = self.inter_trial_interval  # lag between trials
        eff_len = n_trials * (intra_len + lag)

        labels = -torch.ones((n, eff_len, 6), dtype=torch.long)

        for i in range(n):
            # pick digit pair
            pair = random.choice([0, 2, 4, 6, 8])
            digit1, digit2 = pair, pair + 1

            # pick context
            context = random.choice([
                [digit1]*4,
                [digit2]*4,
                [digit2]*3 + [digit1],
                [digit1]*3 + [digit2],
            ])
            oddball_context = {
                tuple([digit1]*4): [digit1]*3+[digit2],
                tuple([digit2]*4): [digit2]*3+[digit1],
                tuple([digit2]*3+[digit1]): [digit2]*4,
                tuple([digit1]*3+[digit2]): [digit1]*4,
            }

            # build trials
            trial_seqs = [context for _ in range(int(n_trials * 0.25))]  # first 5 trials are habituation
            for _ in range(int(n_trials * 0.75)):
                if random.random() < 0.8:
                    trial_seqs.append(context)
                else:
                    trial_seqs.append(oddball_context[tuple(context)])

            # fill labels
            pos = 0
            for t, seq in enumerate(trial_seqs):
                # if t < 5:
                #     continue # skip habituation trials

                blocktype = 0 if context in ([digit1]*4, [digit2]*4) else 1
                local_dev = int(seq[-1] != seq[-2])
                global_dev = int(seq != context)

                for j, s in enumerate(seq):
                    labels[i, pos, 0] = s
                    
                    # stimulus identity
                    if (self.split == "train" and j < 3) or self.split == "test":
                        labels[i, pos, 1] = 0 if s == digit1 else 1
                    # labels[i, pos, 1] = 0 if s == digit1 else 1

                    # serial position
                    if (self.split == "train" and local_dev == 0) or self.split == "test":
                        labels[i, pos, 2] = j
                    
                    # global context
                    if j == 2:
                        labels[i, pos, 3] = blocktype
                    
                    # oddballs
                    if j == 3:
                        labels[i, pos, 4] = local_dev
                        labels[i, pos, 5] = global_dev
                        # if (self.split == "train" and local_dev == 0) or self.split == "test":
                        #     labels[i, pos, 5] = global_dev
                    
                    pos += 2  # 1 step for stimulus + 1 dummy in between
                pos += lag   # add lag between trials

        self.seqlen = eff_len
        return labels


class MNISTSequencesDataset(ImageSequencesDataset):
    """MNIST-specific implementation of image sequence dataset."""
    
    DATASET_DIR = "mnist"
    
    def _load_base_dataset(self, download):
        """Load MNIST dataset."""
        if self.split == "train":
            dataset = MNIST(root=os.path.join(self.root, self.DATASET_DIR), train=True, download=download)
        elif self.split == "test":
            dataset = MNIST(root=os.path.join(self.root, self.DATASET_DIR), train=False, download=download)
        else:
            raise ValueError(f"Split {self.split} not valid. Use one of {self.valid_splits}")
        
        self.dataset = dataset
        self.img_labels = dataset.targets.long()  # shape: (num_samples)
        self.imgs = dataset.data.float() / 255.0  # shape: (num_samples, 28, 28)
        self.imgs = (self.imgs - 0.1307) / 0.3081  # normalize to [0, 1]
        
        # Pre-organize images by class for efficient sampling
        self.imgs_by_class = [self.imgs[self.img_labels == i] for i in range(10)]
    
    def _sample_images_for_sequences(self):
        """Sample MNIST images for the generated sequences."""
        # Initialize data tensor
        h, w = 28, 28  # MNIST image dimensions
        self.data = torch.zeros((self.num_sequences, self.seqlen, h, w), dtype=torch.float32)
        
        # For each sequence
        for i in range(self.num_sequences):
            # For each position in the sequence
            for j in range(self.seqlen):
                # Get the class label
                class_id = self.sequence_sample_labels[i, j].item()

                if class_id == -1:
                    # If the class_id is -1 (used in some sequence types), fill with zeros
                    self.data[i, j] = torch.zeros((h, w), dtype=torch.float32)
                    continue
                
                # Sample a random image of this class
                img_idx = np.random.randint(self.imgs_by_class[class_id].shape[0])
                self.data[i, j] = self.imgs_by_class[class_id][img_idx]


class PFCOddballImgSequencesDataset(ImageSequencesDataset):
    """Implementation of image sequence dataset using images from the original PFC oddball experiment.
    Experiment details: https://doi.org/10.1016/j.celrep.2024.113952
    """
    
    DATASET_DIR = "pfc_oddballs"

    def _load_base_dataset(self, download):
        """Load PFC oddball images from local directory."""
        dataset_path = os.path.join(self.root, self.DATASET_DIR)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"PFC oddball dataset not found in {dataset_path}. Please download it manually.")
        
        # Load images and labels
        self.imgs = []
        self.img_labels = []
        for img_id in range(10):
            img_path = os.path.join(dataset_path, f"resized_imgs_{img_id}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image {img_path} not found in PFC oddball dataset.")
            import matplotlib.pyplot as plt
            img = plt.imread(img_path)  # shape: (H, W, C)
            img = (img - 0.7786261) / 0.3051139  # normalize with mean/std
            self.imgs.append(torch.tensor(img))
            self.img_labels.append(img_id)

        self.imgs = torch.stack(self.imgs)  # shape: (10, H, W, C)
        self.img_labels = torch.tensor(self.img_labels).long()  # shape: (10)

        # Transpose images to (N, C, H, W) format
        self.imgs = self.imgs.permute(0, 3, 1, 2)  # shape: (10, C, H, W)

        # Pre-organize images by class for efficient sampling
        self.imgs_by_class = [self.imgs[i].unsqueeze(0) for i in range(10)]  # each class has one image

    def _sample_images_for_sequences(self):
        """Sample PFC oddball images for the generated sequences."""
        # Initialize data tensor
        c, h, w = self.imgs.shape[1], self.imgs.shape[2], self.imgs.shape[3]  # PFC oddball image dimensions
        self.data = torch.zeros((self.num_sequences, self.seqlen, c, h, w), dtype=torch.float32)
        
        # For each sequence
        for i in range(self.num_sequences):
            # For each position in the sequence
            for j in range(self.seqlen):
                # Get the class label
                class_id = self.sequence_sample_labels[i, j].item()

                if class_id == -1:
                    # If the class_id is -1 (used in some sequence types), fill with zeros
                    self.data[i, j] = torch.zeros((c, h, w), dtype=torch.float32)
                    continue
                
                # Sample the image of this class (only one image per class)
                self.data[i, j] = self.imgs_by_class[class_id][0]

    def __getitem__(self, n):
        x = self.data[n] # shape: (self.seqlen, C, H, W)
        y = self.labels[n] if self.labels is not None else torch.tensor([]) # shape: (num_tasks)
        dense_y = self.dense_labels[n] if self.dense_labels is not None else torch.tensor([]) # shape: (self.seqlen, num_tasks)

        # No need to unsqueeze channel dimension as images already have channels
        return x, (y, dense_y, torch.tensor([]), torch.tensor([]), torch.tensor([])) # cts_y, cts_dense_y, aux_labels are not used in this dataset


class MouseDataset(Dataset):
    """Mouse behavior dataset.
    Videos and DeepLabCut labels are downloaded from:
        https://doi.org/10.6084/m9.figshare.19213272.v1

    Args:
        root (str): The root directory of the dataset
        split (str): The dataset split to use (train/val/test)
        download (bool): Whether to download the dataset (not used here)
        transform (callable, optional): A function/transform to apply to the images
        seq_len (int): The length of the sequences to generate (not used here)
        grayscale (bool): Whether to convert images to grayscale
        seed (int): Random seed for reproducibility
        velocity_stats (dict, optional): Precomputed velocity stats for normalization
        device (str): Device to use for computations
    """
    valid_splits = ["train", "val", "test"]
    DATA_DIR = "mouse_data_96x96"
    NAMES = ["mouse-3-1_96x96", "mouse-2-1_96x96"]
    BODY_PARTS = ["Nose", "Tailroot", "Forehand-Left", "Forehand-Right", "Hindhand-Left", "Hindhand-Right"]

    def __init__(
        self,
        root,
        split="train",
        download=False,
        transform=None,
        seq_len=25,
        grayscale=False,
        seed=42,
        velocity_stats=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.split = split
        self.transform = transform
        self.grayscale = grayscale
        self.seed = seed
        self.device = device

        self.root = os.fspath(root)
        self.data_dir = os.path.join(root, self.DATA_DIR)

        assert split in self.valid_splits, f"Invalid split: {split}. Choose from {self.valid_splits}"
        print("Note: seq_len argument is not used in MouseDataset, fixed to 25 (5 seconds at 5 fps).")

        self.rng = np.random.RandomState(seed)

        self.seq_len = 25  # 5 second chunks of downsampled videos
        self.fps = 25  # original video's fps
        self.downsampled_fps = 5
        self.downscale_factor = self.fps // self.downsampled_fps
        self.seq_len_full = self.seq_len * self.downscale_factor  # 5 seconds of original video
        self.n_total_frames = 90000  # 60 minutes of video at 25 fps

        # set random start indices for sequences in the split
        self.__set_split_start_indices()
        self.num_sequences = len(self.start_indices) * len(self.NAMES)

        # reset labels and dataset
        self.__reset_labels()
        self.__reset_dataset()

        # load data and generate labels
        self._load_data()
        self._to_tensor()
        self.generate_cts_label_keys()
        # discretize the labels
        self.__set_discrete_labels()

        # normalize velocity labels
        if velocity_stats is not None:
            self.__compute_velocity_stats(velocity_stats)
            self.normalize_velocities()

    def __len__(self):
        return self.num_sequences

    def generate_cts_label_keys(self):
        """Generate continuous label keys.
        Used to organize labels for retrieval.
        """
        self.label_keys = [f"{part}_x" for part in self.BODY_PARTS] + \
                           [f"{part}_y" for part in self.BODY_PARTS] + \
                           ["Tailroot_xvel", "Tailroot_yvel"] + \
                           ["sin", "cos"]  # for orientation

    def __getitem__(self, n):
        cts_labels = torch.stack([
            self.labels[k][n] for k in self.label_keys if k not in ["sin", "cos"]
        ], dim=1)
        trig_labels = torch.stack([
            torch.sin(self.labels["orientation"][n]),
            torch.cos(self.labels["orientation"][n])
        ], dim=1)
        cts_labels = torch.cat([cts_labels, trig_labels], dim=1)

        descrt_labels = self.dscrt_labels[n]

        x = self.dataset[n]

        return x, (torch.tensor([]), descrt_labels, torch.tensor([]), cts_labels, torch.tensor([]))
    
    def __set_discrete_labels(self):
        parts = ["Nose", "Tailroot"]
        # get discrete labels
        pos_keys = [f"{part}_x" for part in parts] + [f"{part}_y" for part in parts]
        dscrt_labels_position = torch.stack([
            (np.round(self.labels[k] * 12) % 25).to(int) for k in pos_keys
        ], dim=-1)

        vel_keys = ["Tailroot_xvel", "Tailroot_yvel"]
        positive_direction = torch.stack([
            self.labels[k] > 1e-5 for k in vel_keys
        ], dim=-1)
        negative_direction = torch.stack([
            self.labels[k] < -1e-5 for k in vel_keys
        ], dim=-1)
        direction_labels = positive_direction + 2 * negative_direction

        theta = self.labels["orientation"]
        # get the discrete labels (every 10 degrees)
        theta = torch.round(theta / (np.pi / 18)) % 36
        theta = theta.to(int).unsqueeze(-1)

        self.dscrt_labels = torch.cat([dscrt_labels_position, direction_labels, theta], dim=-1)

    def __set_split_start_indices(self):
        """Set the start indices for sequences in the split.
         For training, we use the first 50 seconds of each one-minute segment.
         For validation/testing, we use the last 10 seconds of each one-minute segment.
        """
        if self.split == "test" and os.path.exists(os.path.join(self.root, "mouse_start_indices_test.npy")):
            # load indices from file
            filename = os.path.join(self.root, "mouse_start_indices_test.npy")
            self.start_indices = np.load(filename).astype(int).tolist()
            return
        # For each one minute segment of the video, we use the first 50 seconds
        # for training and the last 10 seconds for validation/testing.
        self.start_indices = []
        n_segments = self.n_total_frames // (60 * self.fps)  # number of 1-minute segments
        assert n_segments == 60, "Expected 60 segments for 60 minutes of video."
        for segment in range(n_segments):
            segment_start = segment * 60 * self.fps
            if segment == 0:
                print(f"Sampling sequences for {self.split} split...")
            if self.split == "train":
                # first 50 seconds
                idx = self.rng.choice(50 * self.fps - self.seq_len_full + 1,
                                      size=(int((50 * self.fps) / self.downscale_factor),),
                                      replace=False)
                idx += segment_start
                self.start_indices.extend(idx)
            elif self.split in ["val", "test"]:
                # last 10 seconds
                idx = self.rng.choice(10 * self.fps - self.seq_len_full + 1,
                                      size=(int((10 * self.fps) / self.downscale_factor),),
                                      replace=False)
                idx += segment_start + 50 * self.fps
                self.start_indices.extend(idx.tolist())
            if segment == n_segments - 1:
                print("done.")
        if self.split == "test":
            # save indices to file for reproducibility
            filename = os.path.join(self.root, "mouse_start_indices_test.npy")
            np.save(filename, self.start_indices)

    @staticmethod
    def __load_video(name, data_dir, grayscale):
        """Load video frames and fps using torchvision VideoReader.
        
        Args:
            name (str): The name of the video file (without extension)
            data_dir (str): The directory where the video file is located
            grayscale (bool): Whether to convert images to grayscale
        """
        import torchvision
        path = os.path.join(data_dir, name + ".mp4")
        video = torchvision.io.VideoReader(path, "video")
        fps = video.get_metadata()['video']["fps"][0]
        frames = []
        for i, frame in enumerate(video):
            frames.append(frame["data"])
        frames = torch.stack(frames).to(torch.float32) / 255.0
        if grayscale:
            # convert images to grayscale (NTSC formula)
            rgb_mult = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
            frames = torch.sum(frames * rgb_mult, dim=1, keepdim=True)
        return frames, fps

    @staticmethod
    def __load_labels(name, data_dir):
        """Load DeepLabCut labels from CSV file.
        
        Args:
            name (str): The name of the label file (without extension)
            data_dir (str): The directory where the label file is located
        """
        path = os.path.join(data_dir, name + ".csv")
        labels = pd.read_csv(path, header=[0, 1])
        return labels
    
    def __set_positions(self, df, index):
        """Set position labels from DeepLabCut dataframe for a sequence."""
        for part in self.BODY_PARTS:
            self.labels[f"{part}_x"][index] = df[part]["x"].values
            self.labels[f"{part}_y"][index] = df[part]["y"].values
    
    def __set_velocities(self, df, index):
        """Set velocity labels.
         Velocity is computed as the difference in position divided by time.
         A simple smoothing is applied using a moving average over 1 second.
        """
        self.labels["Tailroot_xvel"][index] = df["Tailroot"]["x"].diff().values / self.downsampled_fps
        self.labels["Tailroot_yvel"][index] = df["Tailroot"]["y"].diff().values / self.downsampled_fps

        # shift the velocity labels to align with the position labels
        self.labels["Tailroot_xvel"][index] = np.roll(self.labels["Tailroot_xvel"][index], -1)
        self.labels["Tailroot_yvel"][index] = np.roll(self.labels["Tailroot_yvel"][index], -1)
        # set last value same as the second to last
        self.labels["Tailroot_xvel"][index][-1] = self.labels["Tailroot_xvel"][index][-2]
        self.labels["Tailroot_yvel"][index][-1] = self.labels["Tailroot_yvel"][index][-2]

        # # compute a smoothed velocity over a window of 1 second
        self.labels["Tailroot_xvel"][index] = np.convolve(self.labels["Tailroot_xvel"][index], np.ones(self.downsampled_fps)/self.downsampled_fps, mode='same')
        self.labels["Tailroot_yvel"][index] = np.convolve(self.labels["Tailroot_yvel"][index], np.ones(self.downsampled_fps)/self.downsampled_fps, mode='same')

    def __compute_velocity_stats(self, velocity_stats=None):
        """Compute velocity statistics for normalization.
        If in training mode, compute mean and std from the training data.
        If in validation/testing mode, use provided statistics from train set.
        """
        if self.split == "train":
            self.velocity_stats = {
                "mean": self.labels["Tailroot_xvel"].mean(),
                "std": self.labels["Tailroot_xvel"].std()
            }
        else:
            if velocity_stats is None:
                raise ValueError("velocity_stats must be computed on training data before using them on test/val data.")
            self.velocity_stats = velocity_stats

    def normalize_velocities(self):
        """Normalize velocity labels using precomputed statistics."""
        mean = self.velocity_stats["mean"]
        std = self.velocity_stats["std"]
        self.labels["Tailroot_xvel"] = (self.labels["Tailroot_xvel"] - mean) / (std + 1e-5)
        self.labels["Tailroot_yvel"] = (self.labels["Tailroot_yvel"] - mean) / (std + 1e-5)

    def __set_orientation(self, df, index):
        """Set orientation label based on nose-tail axis.
         Orientation is computed as the angle of the vector from tail to nose.
         In case of missing data (i.e., nose position), the orientation is
         interpolated from the nearest valid positions.
        """
        nose_x = df["Nose"]["x"].values
        nose_y = df["Nose"]["y"].values
        tail_x = df["Tailroot"]["x"].values
        tail_y = df["Tailroot"]["y"].values

        # compute orientation of tail-nose axis
        orientation = np.arctan2(tail_y - nose_y, tail_x - nose_x)

        # set orientation at mask positions to value from the nearest unmasked position
        mask = np.logical_and(nose_x * 96 < 5, nose_y * 96 < 5)
        if mask.any():
            if np.sum(mask) == len(mask):
                orientation[mask] = 0.0
            else:
                orientation[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), orientation[~mask])

        self.labels["orientation"][index] = orientation
    
    def __reset_labels(self):
        """Reset labels to zero arrays."""
        self.labels = {
            f"{part}_x": np.zeros((self.num_sequences, self.seq_len))
            for part in self.BODY_PARTS
        }
        self.labels.update({
            f"{part}_y": np.zeros((self.num_sequences, self.seq_len))
            for part in self.BODY_PARTS
        })
        self.labels.update({
            "Tailroot_xvel": np.zeros((self.num_sequences, self.seq_len))
        })
        self.labels.update({
            "Tailroot_yvel": np.zeros((self.num_sequences, self.seq_len))
        })
        self.labels.update({
            "orientation": np.zeros((self.num_sequences, self.seq_len))
        })
        self.mouse_id = np.zeros(self.num_sequences, dtype=np.int64)

    def _to_tensor(self):
        """Convert dataset and labels to torch tensors."""
        for key in self.labels:
            self.labels[key] = torch.tensor(self.labels[key], dtype=torch.float32)

        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)

    def __reset_dataset(self):
        if self.grayscale:
            self.dataset = np.zeros((self.num_sequences, self.seq_len, 1, 96, 96), dtype=np.float32)
        else:
            self.dataset = np.zeros((self.num_sequences, self.seq_len, 3, 96, 96), dtype=np.float32)

    def _load_data(self):
        """Load video data and labels for all mice and generate sequences."""
        dataset_counter = 0
        for name in self.NAMES:
            position_labels = self.__load_labels(name, self.data_dir)
            video, fps = self.__load_video(name, self.data_dir, self.grayscale)
            assert int(fps) == self.fps, f"FPS mismatch: {fps} != {self.fps}"

            for idx_start in self.start_indices:
                try:
                    self.dataset[dataset_counter] = video[idx_start:idx_start + self.seq_len_full:self.downscale_factor]
                except Exception as e:
                    print(dataset_counter, idx_start, idx_start + self.seq_len_full, len(video))
                    raise e

                if name.startswith("mouse-3"):
                    self.mouse_id[dataset_counter] = 0
                elif name.startswith("mouse-2"):
                    self.mouse_id[dataset_counter] = 1
                labels = position_labels.iloc[idx_start:idx_start + self.seq_len_full:self.downscale_factor]
                self.__set_positions(labels, dataset_counter)
                self.__set_velocities(labels, dataset_counter)
                self.__set_orientation(labels, dataset_counter)
                dataset_counter += 1
