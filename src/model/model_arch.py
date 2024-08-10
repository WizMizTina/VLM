import torch.nn as nn
import torch
import itertools
from torch import optim
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from src.model.factory import *
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPooling
import os
import flash
import re
from contextlib import nullcontext
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning.losses import SupConLoss

from src.grad_cache.pytorch_lightning.pl_gradcache import PLGradCache

os.makedirs('model_cache', exist_ok=True)
os.environ['HF_HOME'] = "model_cache/"

class Patch_Projection(nn.Module):
    def __init__(self, output_emb_size, projection_size):
        super().__init__()
        
        self.linear_projection = nn.Sequential(
            nn.Linear(output_emb_size, projection_size),
        )
        self.non_linear_projection = nn.Sequential(
            nn.Linear(output_emb_size, projection_size),
            nn.GELU(),
            nn.Linear(projection_size, projection_size),
        )
    def forward(self, x):
        return self.linear_projection(x) + self.non_linear_projection(x)


class ImageEncoder(nn.Module):
    def __init__(
        self, image_encoder, patch_alignment: bool = True) -> None:
        super().__init__()


       
        self.model = image_encoder
        self.patch_alignment = patch_alignment
        

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            
            if len(images["pixel_values"].shape) == 5:
                images["pixel_values"] = images["pixel_values"].squeeze(dim=1)
                #print("Image Input shape: ", images["pixel_values"].shape)
                features = self.model.encode_image(images["pixel_values"]) 
                

        else:  
            if len(images.shape) == 5:
                images = images.squeeze(dim=1)
            #print("Image Input shape: ", images.shape)
            features = self.model.encode_image(images) 
            
       
        if isinstance(features, BaseModelOutputWithPooling):
            #print("Base pooler output", features.last_hidden_state.shape)
            features = features.pooler_output
        #if isinstance(features, tuple):  #cls and patch tokens
            #print("Image CLS Output shape: ", features[0].shape)
            #print("Image Patch Output shape: ", features[1].shape)
        #else:
            #print("Image Output shape: ", features.shape)
        return features
        
    
class TextEncoder(nn.Module):
    def __init__(self, text_encoder, patch_alignment: bool = True) -> None:
        super().__init__()


        self.model = text_encoder
        self.patch_alignment = patch_alignment
        
        

    def forward(self, input_ids):
        if not isinstance(input_ids, torch.Tensor):
            #print("Text Input shape: ", input_ids["input_ids"].shape)
            if len(input_ids["input_ids"].shape) == 3:
                input_ids["input_ids"] = input_ids["input_ids"].squeeze(dim =1)
                input_ids["attention_mask"] = input_ids["attention_mask"].squeeze(dim = 1)
            output = self.model.encode_text(input_ids["input_ids"])
            #print("Text Output shape: ", output.shape)
        else:
            #print("Text Input shape: ", input_ids.shape)
            if len(input_ids.shape) == 3:
                input_ids = input_ids.squeeze(dim =1)
            output = self.model.encode_text(input_ids)
            #print("Text Output shape: ", output.shape)
        return output

         
    
    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int,  patch_alignment: bool= True, vision : bool= True) -> None:
        super().__init__()

        self.patch_alignment = patch_alignment
        self.vision = vision
        if not self.patch_alignment:   
            self.projector = None 
        else:
            if self.vision:
                self.vis_projector = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(0.1),
                Patch_Projection(embedding_dim, projection_dim),
                )
            else:
                self.text_projector =  nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim, projection_dim),
                )

            
    def forward(self, x):
       
        if not self.patch_alignment:
            return self.projector(x)
        else:
            if self.vision:
                return self.vis_projector(x)
            else:
                return self.text_projector(x)
        
    

class BaseModel(nn.Module):
    def __init__(self, 
                 channels: int = 12, 
                 patch_alignment: bool = True,
                 base_model_str = "ViT-B-16",
                 ckpt: str = "laion2b-s34b-b88K",
                 clone_weights: bool = True,
                 mean_init : bool = False
                 ):
        super().__init__()
        self.base_model_str = base_model_str
        self.ckpt = ckpt

        if "32" in base_model_str:
            self.stride= 32
        if "14" in base_model_str:
            self.stride = 14
        if "16" in base_model_str:
            self.stride = 16

        self.model,  self.tokenizer = load_model(base_model= base_model_str,ckpt_path=ckpt, channels= channels, clone_weights=clone_weights, mean_init=mean_init)

        if patch_alignment:
            #self.model.visual.positional_embedding = self.interpolate_pos_embed(self.model.visual.positional_embedding.detach(), img_size=224,emb_size = self.model.visual.conv1.out_channels) #not useful for now since we are resizing but might be useful if we do not resize
            self.model.visual.output_tokens = True


        # Move model to device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        

    def interpolate_pos_embed(self, pos_embed, img_size, emb_size):  #TO DO Make shapes dynamic
        
        cls_pos_embed, patch_pos_embed = pos_embed[0,:], pos_embed[1:,:] # torch.Size([768]) torch.Size([196/(49), 768])
        new_num_patches = int(img_size // self.stride) # 14 for B32 and img_size=224 #dynamic from stride in conv1 , 16 for L14
        new_patch_pos_embed = patch_pos_embed.reshape(1, new_num_patches**2, emb_size).transpose(1, 2).reshape(1, emb_size, new_num_patches, new_num_patches) # torch.Size([1, 768, 14, 14]) # torch.Size([1, 768, 7, 7])
        new_patch_pos_embed = torch.nn.functional.interpolate(new_patch_pos_embed, size=(new_num_patches,new_num_patches), mode='bilinear') # torch.Size([1, 768, 25, 25]) torch.Size([1, 768, 7, 7])
        new_patch_pos_embed = new_patch_pos_embed.reshape(1, emb_size, new_num_patches**2).transpose(1,2).squeeze(0) # torch.Size([625, 768])  torch.Size([49, 768])
        new_pos_embed = torch.cat((cls_pos_embed.unsqueeze(0), new_patch_pos_embed),dim=0) # torch.Size([626, 768]) torch.Size([50, 768])
        return torch.nn.Parameter(new_pos_embed)  #torch.Size([50, 768])


        
class ClipLoss(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.logit_scale = 1.0/temperature

    def get_ground_truth(self, device, num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels

    def get_logits(self, image_features, text_features):
        logits_per_image = self.logit_scale * image_features @ text_features.T
        logits_per_text = self.logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return total_loss
        
class CLIPDualEncoderModel(LightningModule):
    def __init__(
        self,
        base_model_str = "ViT-B-32",
        ckpt: str = "laion2b_s34b_b79k",
        patch_alignment : bool = True,
        channels: int = 12,
        warm_up = None,
        max_iter= None,
        initial_temperature: float = 0.07,
        weight_decay: float = 0.05, #penalizes larger weights
        head_lr: float = 3.76e-5,  #projection lr
        clone_weights: bool = True,
        pacl_weight: float = 0.2,
        interval = None,
        frequency = None,
        trainable_modules: list = [],
        full_trainable: bool = False,
        restart: bool = False,
        t_0 : int = None,
        t_multi: int  = None,
        mean_init : bool = False,
        use_gc: bool = True,
        decay_factor: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_alignment = patch_alignment
        self.clip_base_model = BaseModel(patch_alignment=patch_alignment, channels=channels, base_model_str=base_model_str, ckpt=ckpt, clone_weights=clone_weights, mean_init=mean_init)
        self.tokenizer = self.clip_base_model.tokenizer
        self.warm_up = warm_up
        self.max_iter = max_iter
        self.pacl_weight = pacl_weight
        self.interval = interval
        self.frequency = frequency
        self.restart = restart
        self.t_0 = t_0
        self.t_multi = t_multi
        self.head_lr = head_lr
        self.automatic_optimization = (not use_gc) #(not self.params.use_gc) # needed when use_gc is on
        self.fp16 =  use_gc ,#(self.params.precision == 16)
        self.use_gc = use_gc
        self.decay_factor = decay_factor

        if clone_weights:
            trainable_modules.append("visual.conv1.weight")


        print("Settings: ", "Max iter: ", self.max_iter,  " Frequency: ", self.frequency, " Warmup: ", 
        self.warm_up, " Interval: ", self.interval, "Restart scheduler: ", self.restart,  " LR ", self.head_lr,
        " PACL: ",self.patch_alignment, " Unfrozen model: ", full_trainable, " MS Mode: ", clone_weights, " Mean channel init: ", mean_init,
        " Use grad cache: ", self.use_gc, " Decay factor", self.decay_factor)
        print("Trainable: ", trainable_modules)


        #freeze whole model before choosing which modules to unfreeze
        if not full_trainable:
            for param in self.clip_base_model.model.parameters():
                param.requires_grad = False

            #select module to unfreeze
            if trainable_modules is not None:
                for name, param in self.clip_base_model.model.named_parameters():
                    for module in trainable_modules:
                            if module == name:
                                param.requires_grad = True
                 #attention               
                if "attn" in trainable_modules:
                    pattern = re.compile(r"^visual\.transformer\.resblocks\.[012]\.attn")
                    for name, param in self.clip_base_model.model.named_parameters():
                        if pattern.search(name):
                            param.requires_grad = True
        else:
            for param in self.clip_base_model.model.parameters():
                param.requires_grad = True


        
                   
        self.text_encoder = TextEncoder(
            text_encoder= self.clip_base_model.model, 
            patch_alignment=patch_alignment,
        )

        self.image_encoder = ImageEncoder(
            image_encoder = self.clip_base_model.model,
            patch_alignment=patch_alignment,
        )

        self.image_projection = ProjectionHead(
            embedding_dim=  self.clip_base_model.model.visual.positional_embedding.shape[1],
            projection_dim= self.clip_base_model.model.visual.output_dim,
            patch_alignment= patch_alignment
        )
       
        self.text_projection = ProjectionHead(
            embedding_dim=self.clip_base_model.model.token_embedding.embedding_dim,
            projection_dim=self.clip_base_model.model.visual.output_dim,
            patch_alignment= patch_alignment,
            vision = False,
        )

        self.temperature =  nn.Parameter(torch.ones([]) * initial_temperature ) #nn.Parameter(torch.tensor(initial_temperature))
        self.weight_decay = weight_decay

        self.save_hyperparameters()


    #abstract method
    def _compute_losses(self, image_embeddings_patch, image_embeddings_cls, text_embeddings):
        if self.patch_alignment:
            loss_fn = ClipLoss(self.temperature)
            loss_patch = loss_fn(image_embeddings_patch, text_embeddings)
            loss_cls = loss_fn(image_embeddings_cls, text_embeddings)
            return self.pacl_weight*loss_patch + (1-self.pacl_weight)*loss_cls
        else:
            loss_fn = ClipLoss(self.temperature)
            loss_cls= loss_fn(image_embeddings_cls, text_embeddings)
            return loss_cls
        
    def compute_loss(self,image_embeddings_cls, text_embeddings):
        loss_fn = ClipLoss(self.temperature)
        image_embeddings_cls = F.normalize(image_embeddings_cls, dim=-1) 
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        loss_cls= loss_fn(image_embeddings_cls, text_embeddings)
        return loss_cls

    
    def compute_accuracy(self, image_embeddings, text_embeddings):

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        
        
        predicted_texts_indices = logits.argmax(dim=-1)
        predicted_images_indices = logits.T.argmax(dim=-1)
        # Create ground truth indices
        ground_truth_indices = torch.arange(logits.shape[0], device=logits.device)
        
        # Compute the number of correct predictions
        correct_texts = (predicted_texts_indices == ground_truth_indices).float().mean().item()
        correct_images = (predicted_images_indices == ground_truth_indices).float().mean().item()
        # Calculate average accuracy
        accuracy = (correct_texts + correct_images) / 2.0
        
        return accuracy
    
    def accuracy(self, image_embeddings, text_embeddings, topk=(1,)):
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
        
        output = (text_embeddings @ image_embeddings.T) / self.temperature
        target = torch.arange(output.shape[0], device=output.device)

        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        
        pred_T = output.T.topk(max(topk), 1, True, True)[1].t()
        correct_T = pred_T.eq(target.view(1, -1).expand_as(pred_T))

        n = len(target)

        return [(correct[:k].reshape(-1).float().sum(0, keepdim=True) / n).item() for k in topk][0], [(correct_T[:k].reshape(-1).float().sum(0, keepdim=True) / n).item() for k in topk][0]

    def inference_text(self, inputs):
        
        text_features = self.text_encoder(
           inputs
        ) 
        
        return F.normalize(text_features, dim=-1)
    
    def inference_vision(self, image, text_emb=None, retrieval = False, patches = True):
        
        images = self.image_encoder(image)
        

        if isinstance(images, tuple):
            if patches:
                text_emb = self.text_projection(text_emb.permute(1,0))
                image_features = images[1]
                image_embeddings = self.image_projection(image_features)
                print("Shapes of patches: ", text_emb.shape, image_embeddings.shape)
                if text_emb.size(0) != image_embeddings.size(0):
                    image_embeddings = self.pad_tensor_to_match(image_embeddings, text_emb)
                if not retrieval:
                    patch_activations = self.patch_alignment_fn(image_embeddings, text_emb) #torch.Size([500, 196, 10])  number of patches = (img_size/patch_size)**2 + 1, patch size is found in model name i.e 14 for ViT L14"
                    patch_pooled_visual_projections = torch.sum(image_embeddings * patch_activations.unsqueeze(-1), dim=1)
                else:
                    patch_activations = self.patch_alignment_fn(image_embeddings, text_emb, retrieval = retrieval)
                    patch_pooled_visual_projections = torch.sum(image_embeddings.unsqueeze(2) * patch_activations.unsqueeze(3), dim=1) #500 10 512
                    patch_pooled_visual_projections = torch.sum(patch_pooled_visual_projections, dim=1) #500 512
                print("Patched output", patch_pooled_visual_projections.shape)
                return F.normalize(patch_pooled_visual_projections, dim=-1)
            else:
                image_features = images[0]
                return F.normalize(image_features, dim=-1)


        else:
            return  F.normalize(images, dim=-1)

    def forward(self, inputs):
        
        image_features = self.image_encoder(inputs[0])

        text_features = self.text_encoder(
           inputs[1]
        ) 
    
        if isinstance(image_features, tuple):
            image_embeddings = self.image_projection(image_features[1]) #224 224 b32 50 1 49
            text_embeddings = self.text_projection(text_features) 

        else:
            image_embeddings = image_features 
            text_embeddings = text_features  

        if self.patch_alignment:
            patch_activations = self.patch_alignment_fn(image_embeddings, text_embeddings) # shapes =  [B, 196] or 49  weights at patch level
            patch_pooled_visual_projections = torch.sum(image_embeddings * patch_activations.unsqueeze(-1), dim=1) #  (B 196 512) elementwise (B 1 196) then sum = [B, 768] weighted sum across all vision patch embeddings, reduce 196 or 49 to just 1, ie sum along columns to reduce 49 rows to 1 token
            return F.normalize(patch_pooled_visual_projections, dim=-1), F.normalize(image_features[0]), F.normalize(text_embeddings, dim=-1)


        return F.normalize(image_embeddings, dim=-1) , F.normalize(text_embeddings, dim=-1)


    def configure_optimizers(self):
        parameters = [
                #{"params": self.clip_base_model.model.parameters(), "lr": self.head_lr},
                {"params": [param for name, param in self.clip_base_model.model.named_parameters() if "visual.conv1.weight" in name], "lr": self.head_lr},
                {"params": [param for name, param in self.clip_base_model.model.named_parameters() if "visual.conv1.weight" not in name ], "lr": self.head_lr  * self.decay_factor},
                
            ]
        
    
        optimizer = optim.AdamW(parameters, betas=(0.9, 0.90), eps= 1e-6, weight_decay=self.weight_decay)
        if self.restart:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=self.t_0, T_mult=self.t_multi,)
        else:
            lr_scheduler = flash.core.optimizers.LinearWarmupCosineAnnealingLR(optimizer,warmup_epochs=self.warm_up, max_epochs=self.max_iter)
        
        return {
            "optimizer": optimizer,
            'lr_scheduler': {
                'name': 'train/lr',  # put lr inside train group in tensorboard
                'scheduler': lr_scheduler,
                'interval': self.interval, 
                'frequency': self.frequency,
            }
        }

    def init_gc(self, scaler):
        ClipLoss(self.temperature)
        self.trainer.strategy.precision_plugin.forward_context = nullcontext
        self.gc = PLGradCache(
                models=[self.image_encoder, self.text_encoder],
                chunk_sizes=256,
                loss_fn=self.compute_loss,
                fp16=True,
                scaler=(scaler if self.fp16 else None), # needed when using automatic_optimization is off and fp16 is on
                backward_fn=self.manual_backward, # needed when automatic_optimization is off
            )
    def on_train_start(self): # initialize grad cache here
        if self.use_gc:
            self.init_gc(self.trainer.scaler)
           

    def training_step(self, batch, batch_idx, *args, **kwargs):

        if self.use_gc:
            inputs, labels = batch
            assert self.gc is not None
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()
            optimizer.zero_grad()
            loss = self.gc(
                 inputs, labels,
                no_sync_except_last=False,
    
            )
            loss /= max(1, 1) # needed when automatic_optimization is off


            #Check gradients before step
            # for name, param in self.clip_base_model.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.mean()}")
            
            # for param_group in optimizer.param_groups:
            #     print(f"Learning Rate for group: {param_group['lr']}")
            
            # # Check parameter updates after step
            # for name, param in self.clip_base_model.model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter {name} before step: {param.mean()}")
            

            optimizer.step()


            #Check parameter updates after step
            # for name, param in self.clip_base_model.model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Parameter {name} after step: {param.mean()}")

            scheduler.step()
            
        else:
            if self.patch_alignment:
                image_embeddings_patch, image_embeddings_cls, text_embeddings = self.forward(batch)
                loss = self._compute_losses(image_embeddings_patch, image_embeddings_cls, text_embeddings)
            else:
                image_embeddings, text_embeddings = self.forward(batch)
                loss = self._compute_losses(image_embeddings_patch=None, image_embeddings_cls= image_embeddings, text_embeddings=text_embeddings)  

        self.train_loss = self.all_gather(loss).mean()  # remove grad tensor(6.1092, device='cuda:0', grad_fn=<DivBackward0>)
        self.log("train_loss", self.train_loss,prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        if self.patch_alignment:
            image_embeddings_patch, image_embeddings_cls, text_embeddings = self.forward(batch)
            loss = self._compute_losses(image_embeddings_patch, image_embeddings_cls, text_embeddings)
            acc = self.compute_accuracy(image_embeddings_cls, text_embeddings)
            accu_t2i, accu_i2t  = self.accuracy(image_embeddings_cls, text_embeddings)
        else:
            image_embeddings, text_embeddings = self.forward(batch)
            loss = self._compute_losses(image_embeddings_patch=None, image_embeddings_cls= image_embeddings, text_embeddings=text_embeddings)
            acc = self.compute_accuracy(image_embeddings, text_embeddings)
            accu_t2i, accu_i2t  = self.accuracy(image_embeddings, text_embeddings)
        self.val_loss = self.all_gather(loss).mean()
        self.val_acc = self.all_gather(acc).mean()
        self.val_accuracy_t2i =  self.all_gather(accu_t2i).mean()
        self.val_accuracy_i2t =  self.all_gather(accu_i2t).mean()
        self.log(name ="val_loss", value= self.val_loss, prog_bar=True, sync_dist=True)
        self.log(name ="val_acc_custom", value= self.val_acc, prog_bar=True, sync_dist=True)
        self.log(name ="val_acc_i2t", value= self.val_accuracy_i2t, prog_bar=True, sync_dist=True)
        self.log(name ="val_acc_t2i", value= self.val_accuracy_t2i, prog_bar=True, sync_dist=True)
        return loss
    
    def patch_alignment_fn(self, visual_patch_proj, text_cls_proj, retrieval = False): # shapes =  [B, 196, 768], [B, 768] if its 197 or 49 if 50 just do 224/kernel


        
        # normalize text cls token and unsqueeze (required for matmul)
        if not retrieval:
            normalized_visual_patch_proj = F.normalize(visual_patch_proj, dim=-1)
            normalized_visual_patch_proj = normalized_visual_patch_proj.transpose(-2,-1) # shapes =  [B, 768, 196]
            normalized_text_cls_proj = F.normalize(text_cls_proj, dim=-1)
            normalized_text_cls_proj = normalized_text_cls_proj.unsqueeze(1) # shapes =  [B, 1, 768]  

            # compute dot product
            patch_activations = normalized_text_cls_proj @ normalized_visual_patch_proj # shapes =  [B, 1, 196] #patch level similarity
            patch_activations = patch_activations.squeeze() # shapes =  [B, 196]
            # because of dot product, the range is between -1 (least similar) to +1 (most similar)
            # multiply by 10 and apply sigmoid function. this squashes the range from 0 to 1 for every element (not necessarily sums to 1 like that of a softmax function)
        else:
            patch_activations = visual_patch_proj @ text_cls_proj.t()  #B 196 10 for 10 classess
        return F.sigmoid(patch_activations*10)  #weight for each pach
    
    def pad_tensor_to_match(self, tensor, target_tensor):
        # Get the sizes of both tensors
        current_size = tensor.size(0)
        target_size = target_tensor.size(0)
        
        # Compute the difference
        difference = target_size - current_size
        
        # If no padding is needed, return the original tensor
        if difference <= 0:
            return tensor
        
        # Repeat the last element to match the size
        padding = tensor[-1].unsqueeze(0).repeat(difference, *[1]*(tensor.dim()-1))
        
        # Concatenate the padding to the original tensor
        padded_tensor = torch.cat((tensor, padding), dim=0)
        print("Padded tensor shape", padded_tensor.shape)
        
        return padded_tensor
    #abstract method
    def processors(self):
        img_processor, tokenizer = self.clip_base_model.configure_processors()
        return img_processor, tokenizer



