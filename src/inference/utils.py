import torch
import torch.nn as nn

class DictTransforms:
    def __init__(self,
                 dict_transform : dict,
                 ):
        self.dict_transform = dict_transform

    def __call__(self, sample):
        # Apply your transforms to the 'image' key
        for key, function in self.dict_transform.items():
            sample[key] = function(sample[key])
        return sample


class SelectChannels:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels]


class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(dim=self.dim)


class ConvertType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)

class AddMeanChannels:
    """
    Add missing channels to the tensor based on the mean values. Results in zeros after standardization.
    """
    def __init__(self, mean, fill):
        self.mean = mean
        self.mean_tensor = None
        self.zero_tensor = None
        self.fill = fill

    def __call__(self, tensor):
        if self.fill == 'channel_mean' or self.fill == 'channel_drop':
            if self.mean_tensor is None:
                # Init tensor with mean values
                self.mean_tensor = (torch.ones([len(self.mean) - len(tensor), *tensor.shape[1:]]) *
                                    torch.tensor(self.mean)[len(tensor):, None, None])
            # Add mean values for missing channels
            tensor = torch.concat([tensor, self.mean_tensor])
        elif self.fill == 'pixel_mean':
            fill_tensor = tensor.mean(axis=0, keepdim=True).repeat(len(self.mean) - len(tensor), 1, 1)
            tensor = torch.concat([tensor, fill_tensor])
        elif self.fill == 'zero':
            if self.zero_tensor is None:
                self.zero_tensor = torch.zeros_like(tensor.mean(axis=0, keepdim=True)).repeat(len(self.mean) - len(tensor), 1, 1)
            tensor = torch.concat([tensor, self.zero_tensor])

        return tensor

class OneHotEncode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, tensor):
        return torch.nn.functional.one_hot(tensor, self.num_classes)
        
        # tensor = nn.functional.one_hot(torch.tensor(l), self.num_labels)
        # return tensor

        #ForestNet
        #self.num_labels = len(self.label_names)
        #self.labels = nn.functional.one_hot(torch.tensor(self.labels), self.num_labels)