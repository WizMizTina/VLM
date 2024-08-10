from torch.utils.data import Dataset, IterableDataset, get_worker_info, DistributedSampler
from typing import Optional
from typing import Optional
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import LightningDataModule
from torchvision import transforms
import numpy as np
import yaml
import pandas as pd
import os
import xarray as xr
import torch.distributed as dist
import torch


def process_csv(file, clone_weights, tokenizer, file_dir):
    base_filename = os.path.basename(file)
    df = pd.read_csv(file)
    df.set_index(["sample_id", "time_id"], inplace=True)
    df["caption"] = df["caption"].fillna("This is a satellite image")
    parts = os.path.splitext(base_filename)[0].split('_')
    last_part = parts[-1]
    if clone_weights:
        img_path_str = f"{file_dir}/S2L2A/ssl4eo_{last_part}.zarr.zip"
    else:
        img_path_str = f"{file_dir}/S2RGB/ssl4eo_{last_part}.zarr.zip"
    
    return img_path_str, df



class CombinedDataset(Dataset):
    def __init__(self, datasets, captions, tokenizer, clone_weights):
        """
        Args:
            datasets (list of list of tuples): List where each element is a dataset, and each dataset
                                               contains tuples of (images, captions) for each timestamp.
        """
        self.datasets = datasets
        self.captions = captions
        self.tokenizer = tokenizer
        self.clone_weights = clone_weights
        self.num_datasets = len(datasets)
        self.timestamps = 4  # Assuming all datasets have the same number of timestamps
        self.dataset_sizes = 64 # Size of each timestamp
        self.dataset_cache = {}

    def __len__(self):
        """
        Calculate the total number of samples in the combined dataset.
        """
        total_samples = self.dataset_sizes * self.timestamps * self.num_datasets
        return total_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (images, captions) where images and captions are tensors.
        """
        # Determine which timestamp and which dataset to retrieve from
        timestamp_idx = (idx // self.dataset_sizes) % self.timestamps #if 130 timestamp_idx = (130 // 64) % 4 = 2 % 4 = 2
        within_timestamp_idx = idx % self.dataset_sizes  #within_timestamp_idx = 130 % 64 = 2
        
        # Determine which dataset to use
        dataset_idx = (idx // (self.dataset_sizes * self.timestamps)) % self.num_datasets #dataset_idx = (130 // (64 * 4)) % 3 = 130 // 256 % 3 = 0 % 3 = 0
        
         # Check if the dataset is already loaded in the cache
        if dataset_idx not in self.dataset_cache:
            file_path = self.datasets[dataset_idx]
            dataset = xr.open_zarr(file_path, mask_and_scale=False)
            captions = self.captions[dataset_idx]
            self.dataset_cache[dataset_idx] = Xarray_TDataset(dataset, captions, self.tokenizer, self.clone_weights)
        
        dataset = self.dataset_cache[dataset_idx]
        images, captions = dataset[timestamp_idx]
    
        # Retrieve the specific item
        return images[within_timestamp_idx], captions[within_timestamp_idx]
        
      



class Xarray_TDataset(Dataset):
    def __init__(self, data_array, caps_df, tokenizer, clone_weights):
        self.data_array = data_array
        self.captions = caps_df
        self.tokenizer = tokenizer
        self.clone_weights = clone_weights
        
        with open('configs/data_config.yaml', 'r') as file:
            data_params = yaml.safe_load(file)
        
        if self.clone_weights:
            data_params = data_params["ms"]
        else:
            data_params = data_params["rgb"]
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),  # channels last before ToTensor()
            transforms.ToTensor(),  # for rgb the values are scaled but not for ms
            transforms.Normalize(mean=data_params["means"], std=data_params["stds"]),
            transforms.RandomCrop(size=data_params["size"])
        ])

        self.times = self.data_array.coords['time'].values

    def __len__(self):
        return len(self.times)
    
    def __getitem__(self, idx):
        cap_subset = self.captions.loc[self.captions.index.get_level_values('time_id') == idx]
        caps = cap_subset.caption.to_list()
        # Select data for the specific time chunk
        time_chunk = self.data_array.bands.isel(time=idx)
      
        samples = time_chunk.values
        transformed_samples = []

        tokenized_caps = self.tokenizer(caps)

        for i in range(samples.shape[0]):  # Iterate over the sample dimension
            sample = samples[i]
            if self.clone_weights:
                sample = sample.astype(np.float32)  # uint16 cannot be processed
            transformed_sample = self.transform(sample)
            transformed_samples.append(transformed_sample)
            
            # Assuming the captions are indexed the same way as samples
            

        # Stack transformed samples into a single batch
        batch_samples = torch.stack(transformed_samples)
        return batch_samples, tokenized_caps  
                

        
        # Apply transformations and prepare the batch
        


class XarrayDataset(Dataset):
    def __init__(self, data_array, caps, tokenizer, clone_weights):
        self.data_array = data_array
        self.captions = caps
        self.tokenizer = tokenizer
        self.clone_weights = clone_weights
        with open('configs/data_config.yaml', 'r') as file:
            data_params = yaml.safe_load(file)
        if self.clone_weights:
            data_params = data_params["ms"]
        else:
            data_params = data_params["rgb"]

    
        self.transform = transforms.Compose([
        transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))), #channels last before totensor()
        transforms.ToTensor(),  #for rgb the values are scaled but not for ms
        transforms.Normalize(mean= data_params["means"], std=data_params["stds"]),
        transforms.RandomCrop(size=data_params["size"]), #add centre crop for val
        
        ])

    def __len__(self):
        return self.data_array.shape[0]
    
    def __getitem__(self, idx):
        sample = self.data_array[idx]
        if self.clone_weights:
            sample = sample.astype(np.float32)  #uint16 cannot be proccessed
        transformed_sample = self.transform(sample)
        tokenized_cap = self.tokenizer(self.captions[idx])
        return  transformed_sample, tokenized_cap



class CustomIterableDataset(IterableDataset):
    def __init__(self, file_paths,  tokenizer, file_dir, clone_weights, val_mode):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.file_dir = file_dir
        self.clone_weights = clone_weights
        self.val_mode = val_mode
        print("IS MS MODE IN DATA: ", self.clone_weights)

    def process_csv(self, file, i):
        base_filename = os.path.basename(file)
        df = pd.read_csv(file)
        df.set_index(["sample_id", "time_id"], inplace=True)
        df["caption"] = df["caption"].fillna("This is a satellite image")
        parts = os.path.splitext(base_filename)[0].split('_')
        last_part = parts[-1]
        if self.clone_weights:
            img_path_str = f"{self.file_dir}/S2L2A/ssl4eo_{last_part}.zarr.zip"
        else:
            img_path_str = f"{self.file_dir}/S2RGB/ssl4eo_{last_part}.zarr.zip"
        ds = xr.open_zarr(img_path_str, mask_and_scale=False)
        
        #for i in range(4):
        dst = ds.bands.isel(time=i)
        df_subset = df.loc[df.index.get_level_values('time_id') == i]
        caps = df_subset.caption.to_list()
        xarray_dataset = XarrayDataset(dst.values, caps, self.tokenizer, clone_weights=self.clone_weights)
        for j in range(len(xarray_dataset)):
            yield xarray_dataset[j]

    def __iter__(self):
        worker_info = get_worker_info()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if worker_info is None:  # Single-process data loading
            files = self.file_paths
        else:  # Multi-process data loading
            # Split workload among workers
            worker_id = worker_info.id
            total_workers = worker_info.num_workers
            per_worker = int(np.ceil(len(self.file_paths) / float(total_workers)))
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_paths))
            files = self.file_paths[start:end]
        for i in range(4):

            if self.val_mode:
                for file in files:
                    yield from self.process_csv(file,i)
            else:
                for index, file in enumerate(files):
                    if index % world_size == rank:
                        yield from self.process_csv(file, i)

def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        raise ValueError("Batch contains only None values.")
    return torch.utils.data.dataloader.default_collate(batch)

class ImageCapDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer,
        val_split: float = 0.2,
        lazy_loading: bool = False,
        train_batch_size: int = None,
        val_batch_size: int = None,
        num_workers: int = None,
        train_data = None,
        val_data= None,
        trainer = None,
        **kwargs
        ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.val_split = val_split
        self.lazy_loading = lazy_loading
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_data = train_data
        self.val_data = val_data
        self.trainer = trainer
    
    def prepare_data(self):
        pass

    @staticmethod
    def split_data(dataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        train_dataset, val_dataset = random_split(dataset, lengths=[train_length, val_length])
        return train_dataset, val_dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.val_data is None:
            self.train_dataset, self.val_dataset = self.split_data(self.train_data, val_split=self.val_split)
        else:
            self.train_dataset = self.train_data
            self.val_dataset = self.val_data

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers)


