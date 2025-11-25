from collections import defaultdict
import gzip, pickle
import os, subprocess
import torch
import gdown


def write_to_file(data, filename):
    """ Writes an object/dataset to zipped pickle.

    Args:
        data: The (data) object.
        filename: The filename to write to.
    """
    fp = gzip.open("%s"%filename,'wb')
    pickle.dump(data, fp)
    fp.close()


def load_from_file(filename):
    """ Loads an object/dataset from a zipped pickle.
    
    Args:
        filename: The filename to load from.
        
    Returns:
        The loaded object/dataset.
    """
    fp = gzip.open("%s"%filename,'r')
    data = pickle.load(fp)
    fp.close()
    return data
    

def download_librispeech_with_phonemes(data_dir):
    """ Downloads the Librispeech dataset with phonemes from the web.
    
    Args:
        data_dir: Directory to download the dataset to.
    """
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    labels_dir = os.path.join(data_dir, "LibriSpeech100_labels_split")
    if not os.path.isdir(labels_dir):
        os.makedirs(labels_dir)
    
    # Download the dataset
    url = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    filename = os.path.join(data_dir, "train-clean-100.tar.gz")
    root = os.path.join(data_dir, "LibriSpeech")
    if not os.path.exists(filename[:-7]) and not os.path.exists(filename) and not os.path.exists(root):
        print("Downloading Librispeech dataset")
        subprocess.call(["wget", url, "-O", filename])

        # Extract the dataset
        print("Extracting Librispeech dataset...")
        subprocess.call(["tar", "-xzf", filename, "-C", data_dir])

    # Download the splits
    train_split_filename = os.path.join(labels_dir, "train_split.txt")
    if not os.path.exists(train_split_filename):
        print("Downloading train split")
        url = "https://drive.google.com/uc?id=1ubREoLQu47_ZDn39YWv1wvVPe2ZlIZAb"
        gdown.download(url, train_split_filename, quiet=False)

    test_split_filename = os.path.join(labels_dir, "test_split.txt")
    if not os.path.exists(test_split_filename):
        print("Downloading test split")
        url = "https://drive.google.com/uc?id=1vSHmncPsRY7VWWAd_BtoWs9-fQ5cBrEB"
        gdown.download(url, test_split_filename, quiet=False)

    # Download the aligned phoneme labels
    phone_filename = os.path.join(labels_dir, "converted_aligned_phones.zip")
    if not os.path.exists(phone_filename[:-4]) and not os.path.exists(phone_filename):
        print("Downloading aligned phone labels")
        url = "https://drive.google.com/uc?id=1bLuDkapGBERG_VYPS7fNZl5GXsQ9z3p2"
        gdown.download(url, phone_filename, quiet=False)
        subprocess.call(["unzip", phone_filename, "-d", labels_dir])


# Credits: https://github.com/Spijkervet/contrastive-predictive-coding
def libri_flist_reader(flist):
    """ Reads a file list and creates item and speaker dictionaries.
    
    Args:
        flist: Path to the file list.
    
    Returns:
        item_list: List of tuples (speaker_id, dir_id, sample_id).
        speaker_dict: Dictionary mapping speaker IDs to list of item indices.
    """
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


def create_dict_from_phones(phone_path, save_path):
    """ Creates a dictionary mapping sample IDs to phone tensors.

    Args:
        phone_path: Path to the phoneme labels file.
        save_path: Path to save the created dictionary.

    Returns:
        phone_dict: Dictionary mapping sample IDs to phoneme tensors.
    """
    phone_dict = {}
    print("Creating phone dictionary")
    with open(phone_path, "r") as rf:
        for idx, line in enumerate(rf.readlines()):
            tmp = line.replace("\n", "").split(" ")
            sample_id = tmp[0]
            phones = [int(i) for i in tmp[1:]]
            phone_dict[sample_id] = torch.tensor(phones)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path, "wb") as f:
            pickle.dump(phone_dict, f, pickle.HIGHEST_PROTOCOL)
    return phone_dict
