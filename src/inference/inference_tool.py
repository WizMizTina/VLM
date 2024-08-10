#Modified from https://github.com/om-ai-lab/RS5M.git
import logging
import pdb
import tqdm
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import os
from src.inference.classname_and_prompt import *
from src.inference.classname_and_prompt import BigEarth
from src.inference.datasets import RESISC45, EuroSATRGB, AID, EuroSATMS ##
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from src.inference.clip_benchmark.datasets.builder import get_dataset_collate_fn
from src.inference.clip_benchmark.metrics.zeroshot_classification import evaluate
from src.inference.clip_benchmark.metrics.zeroshot_retrieval import recall_at_k, batchify, dataloader_with_indices
from functools import reduce
from datetime import datetime
from torchvision import transforms
import torchmetrics
import copy
import cv2
import numpy as np
import os
import torch
import yaml
from src.inference.datasets.forestnet import  init_forestnet
from src.inference.datasets.bigearthnet import init_bigearthnet



def _convert_to_rgb(image):
    return image.convert('RGB')

# add prprocess for custom model
def get_preprocess(image_resolution=224, is_ms=False, aug=None):

    with open('configs/data_config.yaml', 'r') as file:
        data_params = yaml.safe_load(file)
        

    if is_ms:
        data_params = data_params["ms"]
       
        preprocess_ms = transforms.Compose([
        transforms.Lambda(lambda x: np.delete(x, 10, axis=2)), #only for S2L2A
        transforms.Lambda(lambda x: x.astype(np.float32)),
        
        transforms.ToTensor(),  #for rgb the values are scaled but not for ms
        transforms.Resize(
                size=data_params["size"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        transforms.CenterCrop(data_params["size"]),
        transforms.Normalize(mean= [value - 1000 for value in data_params["means"]], std=data_params["stds"]),
        ])

        


        return preprocess_ms
    else:
        data_params = data_params["rgb"]
        normalize = transforms.Normalize(
            mean=data_params["means"], std=data_params["stds"]
        )
        preprocess_rgb = transforms.Compose([
            transforms.Resize(
                size=data_params["size"],
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(data_params["size"]),
            _convert_to_rgb,
            transforms.ToTensor(),
            normalize,
        ])
        return preprocess_rgb


def zeroshot_get_dataset(dataset_name, root, split, transform=None):

    if dataset_name == "EuroSAT":
        #EuroSAT_root = os.path.join(root)
        EuroSAT_root = os.path.join(root, ".data/eurosat-rgb")

        os.makedirs(EuroSAT_root, exist_ok=True)
        dataset = EuroSATRGB(
            root=EuroSAT_root,
            transform=transform
        )
        dataset.classes =  dataset.classes
        dataset.templates = RSEuroSAT.templates
    
    if dataset_name == "EuroSATMS":
        #EuroSAT_root = os.path.join(root)
        EuroSAT_root = os.path.join(root, ".data")

        os.makedirs(EuroSAT_root, exist_ok=True)
        dataset = EuroSATMS(
            root=EuroSAT_root,
            transform=transform   ########edit
        )
        dataset.classes =  dataset.classes
        dataset.templates = RSMSEuroSAT.templates

    elif dataset_name == "AID":
        AID_root = os.path.join(root, ".data/AID")
        os.makedirs(AID_root, exist_ok=True)
        dataset = AID(
            root=AID_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = RSAID.templates

    elif dataset_name == "RESISC45":
        RESISC45_root = os.path.join(root, ".data/RESISC45/data")
        os.makedirs(RESISC45_root, exist_ok=True)
        dataset = RESISC45(
            root=RESISC45_root,
            transform=transform
        )
        dataset.classes = dataset.classes
        dataset.templates = RSRESISC45.templates

    elif dataset_name == "BigEarthNet":
        bigearthnet_root = os.path.join(root, ".data/ben-ge-8k")
        dataset = init_bigearthnet(bigearthnet_root, [3, 2, 1], True, 19, config_big)
        setattr(dataset, "classes", dataset.class_sets[19])
        setattr(dataset, "templates", BigEarth.templates)
    
    elif dataset_name == "ForestNet":
        forestnet_root = os.path.join(root, ".data/ForestNetDataset")
        dataset = init_forestnet(forestnet_root, False, config_forest_rgb)
        setattr(dataset, "classes", dataset.label_names)
        setattr(dataset, "templates", BigEarth.templates)
        

    if dataset_name not in ["BigEarthNet", "ForestNet"]:
        dataset.classes = [dataset.classes[i].replace('_', ' ') for i in range(len(dataset.classes))]
        dataset.classes = [dataset.classes[i].replace('/', ' ') for i in range(len(dataset.classes))]
        dataset.classes = [dataset.classes[i].lower() for i in range(len(dataset.classes))]

    return dataset


def zeroshot_evaluation(model,  zeroshot_dataset, preprocess, args):
    dataset = zeroshot_get_dataset(dataset_name=zeroshot_dataset, split='test', root=args.test_dataset_dir, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers) #len(dataset.classes)
    logging.info(f'Calculating classifier for {zeroshot_dataset}')
    classnames, prompt_templates = dataset.classes, dataset.templates
    one_class = True
    if zeroshot_dataset== "BigEarthNet":
        one_class = False
  
    tokenizer= open_clip.get_tokenizer(args.model_name)
    classnames = copy.deepcopy(classnames)

    clip_benchmark_metrics = evaluate(model, dataloader, tokenizer, classnames, prompt_templates, args.device, one_class=one_class)
    print(clip_benchmark_metrics)
    return clip_benchmark_metrics



class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", nori_dataset=False,
                 images_dir='bench'):
        logging.debug(f'Loading csv data from {input_filename}.')
        if 'rsicd' in input_filename:
            df = pd.read_csv(input_filename, sep=sep, encoding='gb18030')
        else:
            df = pd.read_csv(input_filename, sep=sep)

        self.nori_dataset = nori_dataset
        self.f = None
        self.images_dir = images_dir

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()

        self.transforms = transforms

        self.duplicate()

        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        texts = self.captions[index]
        image = Image.open(os.path.join(self.images_dir, str(self.images[index])))
        image = self.transforms(image)

        return image, texts

    def duplicate(self):
        unique_images, indexs = np.unique(self.images, return_index=True)
        if len(unique_images) != len(self.images):
            logging.debug(
                f'Amoung all {len(self.images)} images, there are only {len(unique_images)} unique images. Dupication will be performed to enable one-image-to-multiple-text retrieval.')
            self.duplicated_images = []
            self.duplicated_captions = []
            for index in indexs:
                self.duplicated_images.append(self.images[index])
                same_indexs = [i for i, x in enumerate(self.images) if x == self.images[index]]
                captions = []
                for same_index in same_indexs:
                    captions.append(self.captions[same_index])
                self.duplicated_captions.append(captions)

            self.images = self.duplicated_images
            self.captions = self.duplicated_captions






config_big = {
    "model": {
        "name": "CLIP_ViT",
        "img_size": 224,
        "embed_dim": 512,
        "data_mean": [4814.5466, 4578.275, 4082.1073],  #put own values later   scale 
        "data_std": [2686.2954, 2613.0258, 2757.7711]
    },
    "dataset": {
        "name": "BigEarthNetRGB",
        "split": "test"
    },
    "dataloader": {
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": False,
        "shuffle": False
    }
}

config_forest_rgb = {
    "model": {
        "name": "SatMAERGB",
        
        "img_size": 224,       #use own model std mean
        "data_fill": "channel_mean"
    },
    "dataset": {
        "name": "ForestNetRGB",
        "split": "test"
    },
    "dataloader": {
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": False,
        "shuffle": False
    }
}



def get_class_name(filepath):
    filename = os.path.basename(filepath)
    class_name = filename.split('_')[0]
    return class_name

def retrieval_evaluation(model, preprocess, args, recall_k_list=[1, 5, 10, 20], dataset_name=None):

    
    """
    Modified from https://github.com/LAION-AI/CLIP_benchmark/blob/main/clip_benchmark/metrics/zeroshot_retrieval.py
    Evaluate the model on the given dataset

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`

    preprocess: torchvision.transforms
        image preprocessing pipeline

    args: argparse.Namespace
        argument namespace containing required paths and settings

    recall_k_list: list of int
        recall@k k's to use

    dataset_name: str
        name of the dataset

    Returns
    -------

    dict of retrieval metrics
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rmap = torchmetrics.RetrievalMAP()
    

    if dataset_name == "rsitmd":
        dataset = CsvDataset(
            input_filename=os.path.join(args.test_dataset_dir, ".data/RSITMD/RSITMD_img_txt_pairs_test.csv"),
            transforms=preprocess,
            img_key="filepath",
            caption_key="title",
            sep=",",
            images_dir=f"{args.test_dataset_dir}/.data"
        )
    elif dataset_name == "rsicd":
        dataset = CsvDataset(
            input_filename=os.path.join(args.test_dataset_dir, ".data/RSICD/RSICD_img_txt_pairs_test.csv"),
            transforms=preprocess,
            img_key="filepath",
            caption_key="title",
            sep=",",
            images_dir=f"{args.test_dataset_dir}/.data"
        )
    elif dataset_name == "ucmcaptions":
        dataset = CsvDataset(
            input_filename=os.path.join(args.test_dataset_dir, ".data/ucmcaptions/ucmcaptions_img_txt_pairs_test.csv"),
            transforms=preprocess,
            img_key="filepath",
            caption_key="title",
            sep=",",
            images_dir=f"{args.test_dataset_dir}/.data"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=get_dataset_collate_fn('mscoco_captions')
    )

    n_batches = len(dataloader)
    tokenizer = open_clip.tokenize
    # list of batch of images embedding
    batch_images_emb_list = []
    # list of batch of text embedding
    batch_texts_emb_list = []
    # for each text, we collect the corresponding image index, as each image can have multiple corresponding texts
    texts_image_index = []
    dataloader = dataloader_with_indices(dataloader)

    image_paths = []
    for batch_images, batch_texts, inds in tqdm.tqdm(dataloader, total=n_batches):
        batch_images = batch_images.to(args.device)
        # store the index of image for each text
        batch_texts_image_index = [ind for ind, texts in zip(inds, batch_texts) for text in texts]
        # tokenize all texts in the batch
        batch_texts = tokenizer([text for i, texts in enumerate(batch_texts) for text in texts]).to(args.device)
        

        # compute the embedding of images and texts
        with torch.no_grad():
            try:
                batch_text_features = model.encode_text(batch_texts)
                batch_image_features = model.encode_image(batch_images)         
            except AttributeError:
                batch_text_features = model.inference_text(batch_texts)
                batch_image_features = model.inference_vision(batch_images, batch_text_features, retrieval = True)  
            batch_images_emb = F.normalize(batch_image_features, dim=-1)
            batch_texts_emb = F.normalize(batch_text_features, dim=-1)

        batch_images_emb_list.append(batch_images_emb.cpu())
        batch_texts_emb_list.append(batch_texts_emb.cpu())
        texts_image_index.extend(batch_texts_image_index)

        image_paths.extend([dataset.images[i] for i in inds])
    batch_size = len(batch_images_emb_list[0])

    # concatenate all embeddings
    images_emb = torch.cat(batch_images_emb_list)
    texts_emb = torch.cat(batch_texts_emb_list)

    # get the score for each text and image pair
   
    scores = texts_emb @ images_emb.t()

    # Create indexes (each text embedding is its own query group)
    indexes = torch.tensor(texts_image_index, dtype=torch.long)
    indexes = indexes.unsqueeze(1).expand(-1, scores.shape[1])  # Convert to tensor 
    
    # construct the positive pair matrix
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True


    metrics = {}
    metrics_ti = {}
    metrics_it = {}
    for recall_k in recall_k_list:
        metrics[f"retrieval-image2text-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores.T,
                                                                                 positive_pairs.T, batch_size,
                                                                                 args.device,
                                                                                 k=recall_k) > 0).float().mean().item()
        metrics_it[f"retrieval-image2text-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores.T,
                                                                                 positive_pairs.T, batch_size,
                                                                                 args.device,
                                                                                 k=recall_k) > 0).float().mean().item()

    for recall_k in recall_k_list:
        metrics[f"retrieval-text2image-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores, positive_pairs,
                                                                                 batch_size, args.device,
                                                                                 k=recall_k) > 0).float().mean().item()
        metrics_ti[f"retrieval-text2image-R@{recall_k}-{dataset_name}"] = (batchify(recall_at_k, scores, positive_pairs,
                                                                                 batch_size, args.device,
                                                                                 k=recall_k) > 0).float().mean().item()
    

    metrics[f"retrieval-mean-recall-{dataset_name}"] = np.mean(list(metrics.values()))
    metrics[f"retrieval-mean-recall_i2t-{dataset_name}"] = np.mean(list(metrics_it.values()))
    metrics[f"retrieval-mean-recall_t2i-{dataset_name}"] = np.mean(list(metrics_ti.values()))
    

    for key, item in metrics.items():
        metrics[key] = round(float(item), 2)
    
    
    return metrics
