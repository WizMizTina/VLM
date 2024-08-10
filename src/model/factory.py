from huggingface_hub import hf_hub_download
import open_clip
import torch
import torch.nn as nn
import os
from torchvision import transforms
import open_clip
import torch.nn.init as init

os.makedirs('model_cache', exist_ok=True)
os.environ['HF_HOME'] = "model_cache/"

def modify_compose(compose_object, channels: int):
    new_transforms = []
    to_tensor_added = False
    for transform in compose_object.transforms:
        if isinstance(transform, transforms.CenterCrop):
            new_transforms.append(transform)
            #new_transforms.append(transforms.ToTensor())
            #to_tensor_added = True
        elif isinstance(transform, transforms.Normalize):
            #if not to_tensor_added:
                #new_transforms.append(transforms.ToTensor())
                #to_tensor_added = True
            # Extend mean and std for 12 channels
            mean_rgb = transform.mean
            std_rgb = transform.std
            
            # Calculate the mean of the existing 3 values
            mean_existing = sum(mean_rgb) / len(mean_rgb)
            std_existing = sum(std_rgb) / len(std_rgb)
            
            # Extend mean and std to cover 'channels'
            mean_extended = mean_rgb + (mean_existing,) * (channels - len(mean_rgb))
            std_extended = std_rgb + (std_existing,) * (channels - len(std_rgb))
            new_transform = transforms.Normalize(mean=mean_extended, std=std_extended)   
            new_transforms.append(new_transform)
    # Create a new Compose object with modified transforms
    modified_compose = transforms.Compose(new_transforms)
    return modified_compose

def download_from_hub():
    for model_name in ["ViT-L-14", "ViT-B-32", "RN50"]:
        checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", f"RemoteCLIP-{model_name}.pt", cache_dir='model_cache')
        print(f'{model_name} is downloaded to {checkpoint_path}.')


def load_model(base_model: str, clone_weights: bool= True, channels:int = 12, ckpt_path: str = None, mean_init: bool = False):
    
    clip_model, train_img_preprocessor, val_img_preprocessor = open_clip.create_model_and_transforms(base_model, ckpt_path, cache_dir="model_cache")
    state_dict = clip_model.state_dict()
    orig_model = clip_model
    if clone_weights:
        clip_model.visual.conv1 =  nn.Conv2d(
            in_channels=channels, 
            out_channels=orig_model.visual.conv1.out_channels, 
            kernel_size=orig_model.visual.conv1.kernel_size, #dynamically choose
            stride=orig_model.visual.conv1.stride, 
            bias=orig_model.visual.conv1.bias
            )
        
        
        state_dict = extend_weights(state_dict=state_dict,  channels = channels, mean_init= mean_init)
       
        clip_model.load_state_dict(state_dict)  
    
    tokenizer = open_clip.get_tokenizer(base_model)

    return clip_model, tokenizer     

            
def extend_weights(state_dict: dict,  channels: int, mean_init: bool =  True):
    old_patch_weights = state_dict["visual.conv1.weight"]
    out_channels, old_in_channels, kernel_height, kernel_width = old_patch_weights.shape
    new_patch_weights = torch.zeros((out_channels, channels, kernel_height, kernel_width), 
                                device=old_patch_weights.device)
    new_patch_weights[:, 1:2, :, :] = old_patch_weights[:, 2:3, :, :]  # Keep original RGB weights but in BGR format
    new_patch_weights[:, 2:3, :, :] = old_patch_weights[:, 1:2, :, :]  # Keep original RGB weights
    new_patch_weights[:, 3:4, :, :] = old_patch_weights[:, 0:1, :, :]  # Keep original RGB weights
   

    if mean_init:
        mean_weights = torch.mean(old_patch_weights, 1)
        # Add the mean weights with Gaussian noise for channels from 3 to the last one
        std_dev = 0.1
        mean_weights_expanded = mean_weights.unsqueeze(1) 
        noise = torch.randn(mean_weights_expanded.shape, device=old_patch_weights.device) * std_dev  #first band B0
        mean_weights_noise = mean_weights_expanded + noise
        new_patch_weights[:, 0:1, :, :] = mean_weights_expanded

        for c in range(4, channels):
            noise = torch.randn(mean_weights_expanded.shape, device=old_patch_weights.device) * std_dev
            mean_weights_noise = mean_weights_expanded + noise
            new_patch_weights[:, c:c+1, :, :] =mean_weights_expanded
        
    state_dict["visual.conv1.weight"] =  new_patch_weights


    return state_dict

def reverse_weights(state_dict: dict):
    old_patch_weights = state_dict["visual.conv1.weight"]

    permuted_old_patch_weights = old_patch_weights[:, [2, 1, 0], :, :]  #RGB --> BGR
    
    state_dict["visual.conv1.weight"] =  permuted_old_patch_weights


    return state_dict

