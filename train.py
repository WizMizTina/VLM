from src.model.model_arch import CLIPDualEncoderModel
from src.data.data_module import ImageCapDataModule,  CustomIterableDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import argparse
import os
import glob
from datetime import datetime, timedelta
from pytorch_lightning.callbacks import Callback
from src.inference.evaluation import evaluate, evaluate_ms
from src.inference.inference_tool import get_preprocess
import torch
import math
import wandb
import pytorch_lightning as pl
import torch.distributed as dist


class EvaluationCallback(Callback):
    def __init__(self, eval_function, args):
        super().__init__()
        self.eval_function = eval_function
        self.args = args
        self.eval_results = {}  # Store results for later use

    def on_validation_epoch_end(self, trainer, pl_module):
        # Ensure the model is in evaluation mode
        pl_module.eval()
        epoch = trainer.current_epoch
 
        # Call the evaluation function with the model and/or checkpoint path
        eval_result = self.eval_function(pl_module, get_preprocess(image_resolution=224, is_ms=self.args.ms_mode), self.args)
        self.eval_results[epoch] = eval_result
        # Switch back to training mode
        pl_module.train()

class CustomTQDMProgressBar(TQDMProgressBar):
    def __init__(self, total_train_batches, total_val_batches):
        super().__init__()
        self.total_train = total_train_batches
        self.total_val = total_val_batches
        self._is_training = True

    def on_train_start(self, trainer, pl_module):
        self._is_training = True
        super().on_train_start(trainer, pl_module)
        if self.main_progress_bar is not None:
            self.main_progress_bar.total = self.total_train
            self.main_progress_bar.n = 0
            self.main_progress_bar.last_print_n = 0
            self.main_progress_bar.refresh()

    def on_validation_epoch_start(self, trainer, pl_module):
        self._is_training = False
        if self.main_progress_bar is not None:
            self.main_progress_bar.close()  # Close the training progress bar
        super().on_validation_epoch_start(trainer, pl_module)
        if self.main_progress_bar is not None:
            self.main_progress_bar.total = self.total_val
            self.main_progress_bar.n = 0
            self.main_progress_bar.last_print_n = 0
            self.main_progress_bar.refresh()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.main_progress_bar is not None:
            self.main_progress_bar.n = self.main_progress_bar.total
            self.main_progress_bar.refresh()
            self.main_progress_bar.close()
        super().on_validation_epoch_end(trainer, pl_module)
        # Reinitialize the training progress bar for the next epoch
        self._is_training = True

    def on_train_epoch_end(self, trainer, pl_module):
        if self._is_training:
            if self.main_progress_bar is not None:
                self.main_progress_bar.n = self.main_progress_bar.total
                self.main_progress_bar.refresh()
                self.main_progress_bar.close()
            super().on_train_epoch_end(trainer, pl_module)
            self.on_train_start(trainer, pl_module)


def main(args):
    
    pl.seed_everything(42)
    csv_files_train = glob.glob(os.path.join(f"{args.data_dir}/train/captions/", "*.csv"))
    csv_files_val = glob.glob(os.path.join(f"{args.data_dir}/val/captions/", "*.csv"))
    files_to_use = math.ceil(len(csv_files_train)  * args.train_data_percentage)
    csv_files_train = csv_files_train[0: files_to_use]
    print("length of caption datasets")
    print("Train: ", len(csv_files_train))
    print("Val: ", len(csv_files_val))

    # if args.grad_cache:
    #     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #     torch.multiprocessing.set_sharing_strategy("file_system")


    trainable_modules = args.trainable_modules.split(',') if args.trainable_modules else []

    model_rs = CLIPDualEncoderModel(base_model_str=args.base_model, 
                                    ckpt=args.ckpt, 
                                    patch_alignment=args.patch_alignment,
                                    clone_weights=args.ms_mode,
                                    max_iter =  ((len(csv_files_train) * 256)// args.train_batchsize) * args.epochs,
                                    warm_up= args.warmup,
                                    head_lr=args.lr,
                                    pacl_weight=args.pacl_weight,
                                    frequency=args.frequency,
                                    interval=args.interval,
                                    trainable_modules=trainable_modules,
                                    full_trainable=args.unfreeze,
                                    t_0=args.warmup,
                                    t_multi=2,
                                    restart=args.restart,
                                    mean_init=args.mean_init,
                                    decay_factor=args.decay_factor,
                                    use_gc=args.grad_cache


                                    )
    
    tokenizer = model_rs.tokenizer
   
    
    train_dataset = CustomIterableDataset(csv_files_train, tokenizer, f"{args.data_dir}/train",args.ms_mode, val_mode = False)
    val_dataset = CustomIterableDataset(csv_files_val, tokenizer, f"{args.data_dir}/val", args.ms_mode, val_mode = True)
    

    experiment_id = f"lr-{args.lr}-grad_cache-{args.grad_cache}-unfrozen_layers-{args.trainable_modules}-decay_factor_lr-{args.decay_factor}"
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        wandb_logger = WandbLogger(project=args.save_name, name=experiment_id)
    else:
        wandb_logger = None  # Disable logging for non-main processes
    wandb_logger.watch(model_rs, log="all")
    #logger = TensorBoardLogger(save_dir=f"{args.save_path}tensorboard", name="tensorboard_logs")
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint = ModelCheckpoint(save_weights_only=False, dirpath=f"{args.save_path}checkpoints/{args.save_name}/{current_time}/", filename="{epoch}-{step}-{val_loss:.1f}", monitor="val_loss", mode="min", save_on_train_epoch_end=False, save_top_k = -1,every_n_epochs=1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    

    total_train_batches = (len(csv_files_train) * 256) // (args.train_batchsize * args.gpus) #Effective batch size = B X gpus
    print("total train iter", total_train_batches)
    total_val_batches = (len(csv_files_val) * 256) // (args.val_batchsize * args.gpus)
    print("total val iter", total_val_batches )

    # Create the custom progress bar with the total batch sizes
    tqdm = CustomTQDMProgressBar(total_train_batches, total_val_batches)
    

    
    if args.ms_mode:
        eval_callback = EvaluationCallback(evaluate_ms,args)
    else:
        eval_callback = EvaluationCallback(evaluate, args)
            
   
    trainer = Trainer(
            accelerator="gpu",
            devices=args.gpus if args.gpus > 0 else "auto",
            logger=wandb_logger,
            max_epochs=args.epochs,
            log_every_n_steps=1,
            callbacks=[tqdm, lr_monitor, model_checkpoint,eval_callback],
            check_val_every_n_epoch=1,
            gradient_clip_algorithm=args.clip_alg,
            gradient_clip_val= None if args.grad_cache else args.clip_value,
            strategy=args.strategy,
            accumulate_grad_batches= args.k, #reduces steps in tensorboard plot
            precision=args.precision_model,
            limit_train_batches=  total_train_batches,
        
        )
    


    data_module = ImageCapDataModule(
        tokenizer=tokenizer,
        lazy_loading=True,
        train_data=train_dataset,
        val_data=val_dataset,
        train_batch_size = args.train_batchsize,
        val_batch_size= args.val_batchsize,
        num_workers = args.workers,
        trainer=trainer
        
    )

    
    trainer.fit(model_rs, data_module)
    

    # eval_results = eval_callback.eval_results
    # for epoch, results in eval_results.items():
    #     wandb_logger.log_metrics(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/dccstor/geofm-pre/ssl4eos12/",
                        help="dir of data")
    parser.add_argument("--base-model", type=str, default="ViT-B-16",
                        help="base model")
    parser.add_argument("--ckpt", type=str, default="laion2b-s34b-b88K", #laion2b_s34b_b79k
                        help="ckpt path")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs to run")
    parser.add_argument("--strategy", type=str, default="ddp",
                        help="parallel processing strategy")
    parser.add_argument("--lr", type=float, default=3.76e-5,
                        help="learning rate")
    parser.add_argument("--warmup", type=int, default=45,
                        help="warmup iterations")
    parser.add_argument("--pacl-weight", type=float, default=0.2,
                        help="weight of pacl in loss function")
    parser.add_argument("--interval", type=str, default="step",
                        help="step or epoch in lr scheduler")
    parser.add_argument("--frequency", type=int, default=1,
                        help="how often to update lr scheduler")
    parser.add_argument("--clip-alg", type=str, default="norm",
                        help="gradient clipping algorithm")
    parser.add_argument("--clip-value", type=float, default=0.5,
                        help="gradient clipping value")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of workers to use")
    parser.add_argument("--gpus", type=int, default=1,
                        help="number of gpus/ processes per node to use")
    parser.add_argument("--k", type=int, default=1,
                        help="simulate large batches by accumulating k batches before backward step")
    parser.add_argument("--decay-factor", type=float, default=0,
                        help="lr decay factor for later layers")
    parser.add_argument("--train-batchsize", type=int, default=512,
                        help="batch size for training")
    parser.add_argument("--val-batchsize", type=int, default=512,
                        help="batch size for validation")
    parser.add_argument("--unfreeze", type=bool, default=False,
                        help="unfreeze the whole model")
    parser.add_argument("--train-data-percentage", type=float, default=1.0,
                        help="how many files to use from 0 to 1 out of 3400 files")
    parser.add_argument("--patch-alignment", type=bool, default=False,
                        help="whether to do training on all image patches and not just CLS token")
    parser.add_argument('--trainable-modules', type=str, default='visual.proj,text_projection', #,positional_embedding,token_embedding.weight'
                         help='Comma-separated list of trainable module names if model is frozen, no spaces between modules!!!')
    parser.add_argument("--restart", default=False, type=bool,
                        help="use restart in cosine decay scheduler")
    parser.add_argument("--ms-mode", type=bool, default=False,
                        help="MS or rgb")
    parser.add_argument("--precision-model", type=int, default=32,
                        help="precision, for grad cache use 16")
    parser.add_argument("--mean-init", type=bool, default=True,
                        help="initialize extra channels with mean or 0s")
    parser.add_argument("--save-path", type=str, default="",
                        help="where to save checkpoints and results with / at end of path")
    parser.add_argument("--grad-cache", type=bool, default=False,
                        help="use grad cache to simulate large batch size")
    
    #necessary for eval
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
        "--batch-size", default=500, type=int,
        help="batch size",
    )
    
    
    parser.add_argument("--precision", default="amp", type=str)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    current_directory = os.getcwd()

    # Print the current working directory
    print("Current working directory:", current_directory)

    os.makedirs(f'{args.save_path}checkpoints', exist_ok=True)
    os.makedirs(f'{args.save_path}tensorboard', exist_ok=True)

    main(args)
    
