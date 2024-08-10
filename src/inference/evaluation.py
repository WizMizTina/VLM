#Modified from https://github.com/om-ai-lab/RS5M.git
import open_clip
import logging
import torch
import os
import random
import numpy as np
import torch.nn as nn
import argparse
import pytorch_lightning as pl
from src.model.model_arch import CLIPDualEncoderModel
#from old_arch import CLIPDualEncoderModel
from src.inference.inference_tool import (zeroshot_evaluation,
                            retrieval_evaluation,
                            get_preprocess
                            )
from src.inference.classification import evaluate
import csv
from datetime import datetime



def random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False



def build_model(model_name, ckpt_path, device, args):


    preprocess_val = get_preprocess(
        image_resolution=224, is_ms=args.ms
    )
    logging.info(f"model: {model_name}, checkpoint: {ckpt_path}")
    if ckpt_path is None:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=args.pretrained_path,cache_dir="model_cache")
        model.eval()
        
    else :
        model = CLIPDualEncoderModel(base_model_str=args.model_name, ckpt=args.pretrained_path, clone_weights=args.ms).load_from_checkpoint(ckpt_path)
        # if args.ms:
        #     state_dict = model.state_dict()
        #     state_dict["image_encoder.model.conv1.weight"] = state_dict["image_encoder.model.conv1.weight"][:, [3, 2, 1],:,:]
        #     state_dict["clip_base_model.model.visual.conv1.weight"] = state_dict["clip_base_model.model.visual.conv1.weight"][:, [3, 2, 1],:,:]
        #     state_dict["text_encoder.model.visual.conv1.weight"] = state_dict["text_encoder.model.visual.conv1.weight"][:, [3, 2, 1],:,:]

        #     model.image_encoder.model.conv1 =  nn.Conv2d(
        #     in_channels=3, 
        #     out_channels=model.image_encoder.model.conv1.out_channels, 
        #     kernel_size=model.image_encoder.model.conv1.kernel_size, #dynamically choose
        #     stride=model.image_encoder.model.conv1.stride, 
        #     bias=model.image_encoder.model.conv1.bias
        #     )
        #     model.load_state_dict(state_dict)
        model.eval()
        model.freeze()
        
        

    
    
    model = model.to(device)
    print("loaded model")

    

    return model, preprocess_val

def write_to_csv(metrics_dict, dataset, model_name, args):

    os.makedirs(f'{args.save_path}bench_results_custom/{args.save_name}', exist_ok=True)
    os.makedirs(f'{args.save_path}bench_results/{args.save_name}', exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.custom_model:
        csv_file = f'{args.save_path}bench_results_custom/{args.save_name}/benchmark_{dataset}_{current_time}.csv'
    else:
         csv_file = f'{args.save_path}bench_results/benchmark_{dataset}_{current_time}_{args.save_name}.csv'
   
    # Extract keys and values as rows
    rows = [(key, value) for key, value in metrics_dict.items()]

    # Write to CSV
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print(f'Dictionary data has been written to {csv_file}.')


def evaluate(model, preprocess, args):
    print("making val dataset with transformation: ")
    print(preprocess)
    zeroshot_datasets = [
        "EuroSAT",
        "ForestNet",
        "RESISC45",
        "AID",
        "BigEarthNet"
    ]
    
    model.eval()
    all_metrics = {}

    # zeroshot classification
    metrics_cl = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zeroshot_evaluation(model, zeroshot_dataset, preprocess, args)
        write_to_csv(zeroshot_metrics, zeroshot_dataset, args.model_name, args)
        metrics_cl.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)
        
    

    #retrieval
    metrics_ret = {}
    retrival_datasets = [
       "rsitmd", "rsicd", "ucmcaptions"
        ]
    for dataset in retrival_datasets:
        retrieval_metrics = retrieval_evaluation(model, preprocess, args, recall_k_list=[1, 5, 10],
                                                   dataset_name=dataset)
        write_to_csv(retrieval_metrics, dataset, args.model_name, args)
        #metrics_ret.update(retrieval_metrics)
        #all_metrics.update(retrieval_metrics)
    
    

    return all_metrics

def evaluate_ms(model, preprocess, args):
    print("making val dataset with transformation: ")
    print(preprocess)
    zeroshot_datasets = [
        "EuroSATMS",
        #"ForestNet",
        #"BigEarthNet"
    ]
    
    model.eval()
    all_metrics = {}

    # zeroshot classification
    metrics_cl = {}
    for zeroshot_dataset in zeroshot_datasets:
        zeroshot_metrics = zeroshot_evaluation(model, zeroshot_dataset, preprocess, args)
        write_to_csv(zeroshot_metrics, zeroshot_dataset, args.model_name, args)
        metrics_cl.update(zeroshot_metrics)
        all_metrics.update(zeroshot_metrics)
        

    
    

    return all_metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default="ViT-B-16", type=str,
        help="ViT-B-32 or ViT-L-14",
    )
    parser.add_argument(
        "--pretrained-path", default="laion2b_s34b_b88k", type=str,
        help=" Pretrained",
    )
    parser.add_argument(
        "--save-name", default="ms", type=str,
        help="unique saving name for results csv",
    )
    parser.add_argument(
        "--ckpt-path", default=None, type=str,
        help="Path ckpt.pt file",
    )
    parser.add_argument(
        "--ms", default=False, type=bool,
        help="RGB or MS",
    )
    
    parser.add_argument(
        "--subset", default="clip", type=str,
        help="subset of mean and std",
    )
    parser.add_argument(
        "--custom-model", default=True, type=bool,
        help="custom model or base model",
    )
    parser.add_argument(
        "--random-seed", default=3407, type=int,
        help="random seed",
    )
    parser.add_argument(
        "--test-dataset-dir", default="bench", type=str,
        help="test dataset dir",
    )
    parser.add_argument(
        "--batch-size", default=100, type=int,
        help="batch size",
    )
    parser.add_argument(
        "--workers", default=8, type=int,
        help="number of workers",)
    parser.add_argument("--precision", default="amp", type=str)
    parser.add_argument("--save-path", type=str, default="",
                help="where to save checkpoints and results")
    
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    # random_seed(args.random_seed)
    logging.basicConfig(
        level=os.getenv('log_level', 'INFO'),
        handlers=[logging.StreamHandler(), logging.FileHandler("log_evaluation")],  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    #ckpt_paths = [os.path.join("checkpoints/Unfrozen", ckpt) for ckpt in os.listdir("checkpoints/Unfrozen") if ckpt.endswith('.ckpt')]
    model, img_preprocess = build_model(args.model_name, args.ckpt_path, args.device, args)
    if not args.ms:
        eval_result = evaluate(model, img_preprocess, args)
    else:
        val_result = evaluate_ms(model, img_preprocess, args)
    # for ckpt_path in ckpt_paths:
    #     model, img_preprocess = build_model(args.model_name, ckpt_path, args.device, args)
    #     eval_result = evaluate(model, img_preprocess, args)
    #     #write_to_csv(eval_result, ckpt_path)

    # for key, value in eval_result.items():
    #     print("{}: {}".format(key, value))


if __name__ == "__main__":
    main()
