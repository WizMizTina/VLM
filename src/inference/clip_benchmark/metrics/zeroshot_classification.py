"""
Code adapated from https://github.com/mlfoundations/open_clip/blob/main/src/training/zero_shot.py
Thanks to the authors of OpenCLIP
"""
import logging
from contextlib import suppress

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import RetrievalMAP, RetrievalHitRate
import numpy as np
from sklearn.metrics import classification_report, balanced_accuracy_score
from collections import defaultdict


def zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=True):
    """
    This function returns zero-shot vectors for each class in order
    to use it for zero-shot classification.
    

    model:
        CLIP-like model with `encode_text`
    
    tokenizer:
        text tokenizer, i.e. convert list of strings to torch.Tensor of integers
    
    classnames: list of str
        name of classes
    
    templates: list of str
        templates to use.
    
    Returns
    -------
    
    torch.Tensor of shape (N,C) where N is the number
    of templates, and C is the number of classes.
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    with torch.no_grad(), autocast():
        zeroshot_weights = []
        emb = []
        for classname in tqdm(classnames):
            if type(templates) == dict:
                # class-specific prompts (e.g., CuPL https://arxiv.org/abs/2209.03320)
                texts = templates[classname]
            elif type(templates) == list:
                # generic prompts tht are specialized for each class by replacing {c} with the class name
                texts = [template.format(c=classname) for template in templates]
            else:
                raise ValueError("templates must be a list or a dict")
            texts = tokenizer(texts).to(device)  # tokenize
            try:
                class_embeddings = model.encode_text(texts)
            except AttributeError:
                class_embeddings = model.inference_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm() 
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
       
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy

    output: torch.Tensor
        shape (N, C) where N is the number of examples, C the number of classes.
        these are the logits.
    
    target: torch.Tensor
        shape (N,) where N is the number of examples. Groundtruth class id of each example.
    
    topk: tuple
        which topk to compute, e.g., topk=(1,5) will compute top-1 and top-5 accuracies
    
    Returns
    -------
    
    list of top-k accuracies in the same order as `topk`
    """
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) / n for k in topk]


def run_classification(model, classifier, dataloader, device, amp=True, one_class = True):
    """
    Run zero-shot classifcation

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    classifier: torch.Tensor
        obtained from the function `zero_shot_classifier`
    
    dataloader: torch.utils.data.Dataloader 
    
    Returns
    -------
    (pred, true)  where
        - pred (N, C) are the logits
        - true (N,) are the actual classes
    """
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred = []
    true = []
    nb = 0
    class_correct = defaultdict(int)  # ADDED 0 (default value provided by int())
    class_total = defaultdict(int) 
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            target = target.to(device)

            with autocast():
                # predict
                try:
                    image_features = model.encode_image(images)
                    if isinstance(image_features, tuple):
                        image_features = image_features[0]
                except AttributeError:
                    image_features = model.inference_vision(images, classifier, patches =   False) #B 10 (Eurosat) 512, classifier is 512, 10
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier
            
            true.append(target.cpu())
            pred.append(logits.float().cpu())

            if one_class:
                _, preds = torch.max(logits, 1)  # ADDED
                for label, predx in zip(target, preds):  # ADDED
                    if label == predx:  # ADDED
                        class_correct[label.item()] += 1  # ADDED
                    class_total[label.item()] += 1  # ADDED

    pred = torch.cat(pred)
    true = torch.cat(true)
    if one_class:
        class_accuracies = {classname: class_correct[i] / class_total[i] for i, classname in enumerate(dataloader.dataset.classes)}  # ADDED
        return pred, true, class_accuracies
    else:
        return pred, true
    
def map_per_class(scores, targets, topk=100):
    map_per_class = {}
    for k in range(scores.size(1)):
        scores_k = scores[:, k]
        cls_indexes = torch.zeros(scores.size(0), dtype=torch.long)
        cls_relevance = (targets == k)
        rmap = RetrievalMAP(top_k = topk)
        cls_map = rmap(preds= scores_k, target= cls_relevance, indexes= cls_indexes)
        map_per_class[k] = cls_map.item()
    return map_per_class

def map_per_class_ml(scores, targets,topk = 100):
    map_per_class = {}
    for k in range(scores.size(1)):
        scores_k = scores[:, k]
        cls_relevance = targets[:, k]
        cls_indexes = torch.zeros(scores.size(0), dtype=torch.long)
        rmap = RetrievalMAP(top_k=topk)
        cls_map = rmap(preds=scores_k, target=cls_relevance, indexes=cls_indexes)
        map_per_class[k] = cls_map.item()
    return map_per_class


def average_precision_per_class(scores, targets):

    """
    Compute average precision  for each class
    this metric is used for multi-label classification
    see explanations here https://fangdahan.medium.com/calculate-mean-average-precision-map-for-multi-label-classification-b082679d31be
    Code is adapted from https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py, thanks to the authors of `tnt`.

    Parameters
    ----------

    scores: torch.Tensor
        logits, of shape (N,C) where N is the number of examples, C the number of classes
    
    targets: torch.Tensor
        one-hot vectors of groundtruth targets (N, C), where N is the number of examples, C is the
        number of classes
    
    Returns
    -------

    torch.Tensor of shape (C,) of avereage precision for each class, where C is     
    the number of classes.
    
    """
    ap = torch.zeros(scores.size(1))
    rg = torch.arange(1, scores.size(0) + 1).float()
    # compute average precision for each class
    for k in range(scores.size(1)):
        # sort scores
        scores_k = scores[:, k]
        targets_k = targets[:, k]
        _, sortind = torch.sort(scores_k, 0, True)
        truth = targets_k[sortind]
        tp = truth.float().cumsum(0)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap[k] = precision[truth.bool()].sum() / max(float(truth.sum()), 1)
    return ap

def multilabel_accuracy(true_labels, predicted_scores):
    """
    Calculate the  accuracy for multi-label classification where k is the number of true labels for each sample.
    
    Parameters:
        true_labels (torch.Tensor): Binary matrix of shape (n_samples, n_classes)
        predicted_scores (torch.Tensor): Matrix of predicted scores of shape (n_samples, n_classes)
    
    Returns:
        float: The top-k accuracy of the model
    """
    num_samples = true_labels.size(0)
    num_classes = true_labels.size(1)
    total_accuracy = 0.0
    class_correct_counts = torch.zeros(num_classes)
    class_total_counts = torch.zeros(num_classes)
    class_acc = {}
    
    for i in range(num_samples):
        # Get the true labels for the current sample
        true_label_indices = true_labels[i].nonzero(as_tuple=True)[0]
        k = len(true_label_indices)  # Number of true labels
        
        
        if k == 0: #assumption is every image has label
            continue
        
        # Get the indices of the top-k predicted scores
        top_k_values, top_k_indices = torch.topk(predicted_scores[i], k)
        
        
        # Calculate the number of true labels that are in the top-k predictions
        num_correct_labels = sum(1 for label in true_label_indices if label in top_k_indices)
        
        
        # Calculate the partial accuracy score for this sample
        partial_accuracy = num_correct_labels / k
        total_accuracy += partial_accuracy

        # Update counters for per-class accuracy
        for label in true_label_indices:
            class_total_counts[label] += 1
            if label in top_k_indices:
                class_correct_counts[label] += 1
    
    # Compute average partial top-k accuracy
    average_top_k_acc = total_accuracy / num_samples
    per_class_accuracy = class_correct_counts / class_total_counts
    return average_top_k_acc, per_class_accuracy


def evaluate(model, dataloader, tokenizer, classnames, templates, device, amp=True, verbose=True, save_clf=None, load_clfs=[], one_class = True):
    """
    Run zero-shot classification and evaluate the metrics

    Parameters
    ----------

    model: torch.nn.Module
        CLIP-like model with `encode_image` and `encode_text`
    
    dataloader: torch.utils.data.Dataloader

    tokenizer: text tokenizer

    classnames: list of str
        class names
    
    templates: list of str
        templates to use for zero-shot classification
    
    device: cpu/cuda

    amp: whether to use automatic mixed precision

    verbose: whether to use verbose model

    Returns
    -------

    dict of classification metrics
    """
    if len(load_clfs) > 0:
        n = len(load_clfs)
        classifier = torch.load(load_clfs[0], map_location='cpu') / n
        for i in range(1, n):
            classifier = classifier + torch.load(load_clfs[i], map_location='cpu') / n
        classifier = classifier.to(device)
    else:
        classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device, amp=amp)
    
    if save_clf is not None:
        torch.save(classifier, save_clf)
        # exit() - not sure if we want to exit here or not.

    
    
    if one_class:
        logits, target, class_accuracies = run_classification(model, classifier, dataloader, device, amp=amp)
    else:
        logits, target = run_classification(model, classifier, dataloader, device, amp=amp, one_class = False)
    is_multilabel = (len(target.shape) == 2)
    cls_map = {}
    cls_acc = {}
    cls_rmap = {}
    if is_multilabel:
        if verbose:
            print("Detected a multi-label classification dataset")
        multi_acc,  per_class_acc  = multilabel_accuracy(target, logits)
        # Multiple labels per image, multiple classes on the dataset
        ap_per_class= average_precision_per_class(logits, target)    #manual calculation of rmAP
        mAP = map_per_class_ml(logits, target, topk=100) #multilabel
        for class_name, acc in zip(dataloader.dataset.classes, per_class_acc.tolist()):
                print(f"Class: {class_name}, Accuracy: {acc}")
                cls_acc[f"{class_name}_acc"] = acc
        cls_acc["Average_class_accuracy"] = np.mean(list(per_class_acc))
        for class_name, rmap in zip(dataloader.dataset.classes, mAP.values()):
                print(f"Class: {class_name}, mAP: {rmap}")
                cls_rmap[class_name] = rmap
        cls_rmap["Ave_mAP"] =  np.mean(list(mAP.values()))
        if verbose:
            for class_name, ap in zip(dataloader.dataset.classes, ap_per_class.tolist()):
                print(f"Class: {class_name}, AveragePrecision: {ap}")
                cls_map[class_name] = ap

        return {"mean_average_precision": ap_per_class.mean().item(), "class map": cls_map, "multilabel accuracy:": multi_acc,"class acc": cls_acc, "rmAP": cls_rmap}
    else:
        map_cls = map_per_class(logits, target, topk=100)
        for class_name, ap in zip(dataloader.dataset.classes,  map_cls.values()):
                print(f"Class: {class_name}, mAP: {ap}")
                cls_map[f"{class_name}_maP"] = ap
        cls_map["mAP"] =  np.mean(list(map_cls.values()))

        # Single label per image, multiple classes on the dataset
        # just compute accuracy and mean_per_class_recall

        pred = logits.argmax(axis=1)
        # measure accuracy
        
       
        if len(dataloader.dataset.classes) >= 5:
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        else:
            acc1, = accuracy(logits, target, topk=(1,))
            acc5 = float("nan") 
        mean_per_class_recall = balanced_accuracy_score(target, pred)
        if verbose:
            print(classification_report(target, pred, digits=3))
        return {"acc1": acc1, "acc5": acc5, "mean_per_class_recall": mean_per_class_recall, **{key: value for key, value in class_accuracies.items()},**{key: value for key, value in cls_map.items()} }
