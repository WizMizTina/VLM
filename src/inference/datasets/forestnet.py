
import os
import h5py
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, io
import torch.nn.functional as F
from src.inference.utils import Unsqueeze, SelectChannels, AddMeanChannels
from src.inference.utils import SelectChannels, Unsqueeze, AddMeanChannels


LABEL_NAMES = ['Oil palm plantation', 'Timber plantation', 'Other large-scale plantations', 'Grassland shrubland', 'Small-scale agriculture', 'Small-scale mixed plantation', 'Small-scale oil palm plantation', 'Mining', 'Fish pond', 'Logging', 'Secondary forest', 'Other']
LABEL_DICT = {'Oil palm plantation': 0, 'Timber plantation': 1, 'Other large-scale plantations': 2, 'Grassland shrubland': 3, 'Small-scale agriculture': 4, 'Small-scale mixed plantation': 5, 'Small-scale oil palm plantation': 6, 'Mining': 7, 'Fish pond': 8, 'Logging': 9, 'Secondary forest': 10, 'Other': 11}
LABEL_NAMES_MERGED = ['Plantation', 'Grassland shrubland', 'Smallholder agriculture', 'Other']
MERGED_LABEL_DICT = {'Plantation': 0, 'Grassland shrubland': 1, 'Smallholder agriculture': 2, 'Other': 3}

MERGE_LABEL_DICT = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3}


class ForestNet(Dataset):
    """
    Dataset class for ForestNet, introduced in:
    Irvin, J., Sheng, H., Ramachandran, N., Johnson-Yu, S., Zhou, S., Story, K., ... & Ng, A. Y. (2020).
    Forestnet: Classifying drivers of deforestation in indonesia using deep learning on satellite imagery.
    arXiv preprint arXiv:2011.05479. https://arxiv.org/pdf/2011.05479.pdf
    """
    def __init__(self, dataset_root, split, transform=None, merge_labels=False):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform

        assert os.path.isdir(dataset_root), ('ForestNet data not found. '
                                             'Download dataset with `sh datasets/forestnet_download.sh`')

        # Load file names and labels
        split_df = pd.read_csv(os.path.join(dataset_root, f'{split}.csv'))
        self.filenames = split_df['example_path'].values

        if not merge_labels:
            self.labels = split_df['label'].values
            self.labels = [LABEL_DICT[l] for l in self.labels]
            self.label_names = LABEL_NAMES
        else:
            self.labels = split_df['merged_label'].values
            self.labels = [MERGED_LABEL_DICT[l] for l in self.labels]
            self.label_names = LABEL_NAMES_MERGED

        self.num_labels = len(self.label_names)
        self.labels_list = self.labels
        self.labels = F.one_hot(torch.tensor(self.labels), self.num_labels)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        # Load RGB and infrared data from png and npy file using the composite data
        visible = io.read_image(os.path.join(self.dataset_root, self.filenames[index], 'images', 'visible', 'composite.png'))
        # Stack data into the expected order
        data = visible
        

        if self.transform is not None:
            data = self.transform(data)


        sample = {
            'image': data,
            'label': torch.tensor(self.labels_list[index])
        }
        if sample['image'].dim() == 4 and sample['image'].size(1) == 1:
            sample['image'] = sample['image'].squeeze(1)
        return sample['image'], sample['label']





def init_forestnet(forestnet_dir, merge_labels, cfg, *args, **kwargs):
    """
    Init m-ForestNet dataset.
    """
    # Get dataset parameters
    split = cfg['dataset']['split']

    

    # Init transforms
    normalize_img = transforms.Normalize(
            mean=[0.48145466*127.5, 0.4578275*127.5, 0.40821073*127.5], std=[0.26862954*127.5, 0.26130258*127.5, 0.27577711*127.5])
    image_transforms = [
        transforms.Resize(size=cfg['model']['img_size'], antialias=True),  
        #transforms.Lambda(lambda x: x.numpy()),
        #transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),
        normalize_img
    ]

    # Init dataset
    dataset = ForestNet(
        dataset_root=forestnet_dir,
        split=split,
        transform=transforms.Compose(image_transforms),
        merge_labels=merge_labels,
    )

    return dataset



