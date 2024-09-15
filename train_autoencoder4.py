import torch
from ldm.util import instantiate_from_config
import numpy as np
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
import lpips
from tqdm import tqdm
from distributed import get_rank, synchronize
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #


def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["autoencoder"] = {}

    for k,v in state_dict.items():
        if k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v    
    return out 


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def sub_batch(batch, num=1):
    # choose first num in given batch 
    num = num if num > 1 else 1 
    for k in batch:
        batch[k] = batch[k][0:num]
    return batch


def wrap_loader(loader):
    while True:
        for batch in loader:
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

           
def create_expt_folder_with_auto_resuming(OUTPUT_ROOT, name):
    name = os.path.join( OUTPUT_ROOT, name )
    writer = None
    checkpoint = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'autoencoder_checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming autoencoder ckpt found '+ potential_ckpt)
                break 
        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 


class Trainer_Autoencoder:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )

        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.opt = torch.optim.AdamW(self.autoencoder.parameters(), lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(self.autoencoder.parameters())
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()

        # = = = = = = = = = = = = = = = = = = = = create scheduler = = = = = = = = = = = = = = = = = = = = #
        if config.scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps, num_training_steps=config.total_iters)
        elif config.scheduler_type == "constant":
            self.scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=config.warmup_steps)
        else:
            assert False 

        # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None  # None
        dataset_train = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=True, repeats=train_dataset_repeats)
        sampler = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)     

        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            if self.starting_iter >= config.total_iters:
                print("Training finished. Start exiting")
                exit()
        else:
            load_key, no_load_key, temp_dict = [], [], {}
            model_dict      = self.autoencoder.state_dict()
            state_dict_sd = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )["autoencoder"]
            for k, v in state_dict_sd.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                elif k.replace('decoder', 'decoder_1') in model_dict.keys() and np.shape(model_dict[k.replace('decoder', 'decoder_1')]) == np.shape(v):
                    temp_dict[k.replace('decoder', 'decoder_1')] = v
                    temp_dict[k.replace('decoder', 'decoder_2')] = v
                    temp_dict[k.replace('decoder', 'decoder_3')] = v
                    temp_dict[k.replace('decoder', 'decoder_4')] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            self.autoencoder.load_state_dict(model_dict)

            if get_rank() == 0:
                print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
                print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    def run_one_step(self, batch):
        z = self.autoencoder.encode( batch["image_VIS"], batch["image_NIR"], batch["image_SWIR"], batch["image_THERMAL"] )
        dec_VIS, dec_NIR, dec_SWIR, dec_THERMAL = self.autoencoder.decode(z)
        loss_mse = (torch.nn.functional.mse_loss(dec_VIS, batch["image_VIS"]) + torch.nn.functional.mse_loss(dec_NIR, batch["image_NIR"]) + torch.nn.functional.mse_loss(dec_SWIR, batch["image_SWIR"]) + torch.nn.functional.mse_loss(dec_THERMAL, batch["image_THERMAL"]))
        loss_lpips = (self.lpips_loss(dec_VIS, batch["image_VIS"]) + self.lpips_loss(dec_NIR, batch["image_NIR"]) + self.lpips_loss(dec_SWIR, batch["image_SWIR"]) + self.lpips_loss(dec_THERMAL, batch["image_THERMAL"])).mean()
        loss = loss_mse + loss_lpips
        self.loss_dict = {"loss": loss.item(), "loss_mse": loss_mse.item(), "loss_lpips": loss_lpips}

        return loss
        

    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.autoencoder.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx
            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            loss.backward()
            self.opt.step() 
            self.scheduler.step()

            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()

        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):
        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            # Do an inference on one training batch 
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_train), batch_here)
            batch_to_device(batch, self.device)

            z = self.autoencoder.encode( batch["image_VIS"], batch["image_NIR"], batch["image_SWIR"], batch["image_THERMAL"] )
            dec_VIS, dec_NIR, dec_SWIR, dec_THERMAL = self.autoencoder.decode(z)

            save_path = os.path.join(self.name, str(iter_name).zfill(8)+'.png')
            torchvision.utils.save_image( torch.cat([batch["image_VIS"], batch["image_NIR"], batch["image_SWIR"], batch["image_THERMAL"], dec_VIS, dec_NIR, dec_SWIR, dec_THERMAL], dim=0), save_path, nrow=self.config.batch_size * 4, normalize=True, scale_each=True, range=(-1,1) )

        ckpt = dict(
            autoencoder = self.autoencoder.state_dict(),
            opt = self.opt.state_dict(),
            scheduler= self.scheduler.state_dict(),
            iters = self.iter_idx+1,
            config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "autoencoder_checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "autoencoder_checkpoint_latest.pth") )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    import torch.multiprocessing as multiprocessing

    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_ROOT", type=str,  default="DATA", help="path to DATA")
    parser.add_argument("--OUTPUT_ROOT", type=str,  default="OUTPUT", help="path to OUTPUT")

    parser.add_argument("--name", type=str,  default="test", help="experiment will be stored in OUTPUT_ROOT/name")
    parser.add_argument("--seed", type=int,  default=123, help="used in sampler")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--yaml_file", type=str,  default="configs/flickr.yaml", help="paths to base configs.")

    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=2, help="")
    parser.add_argument("--workers", type=int,  default=1, help="")
    parser.add_argument("--official_ckpt_name", type=str,  default="sd-v1-4.ckpt", help="SD ckpt name and it is expected in DATA_ROOT, thus DATA_ROOT/official_ckpt_name must exists")
    parser.add_argument("--ckpt", type=lambda x:x if type(x) == str and x.lower() != "none" else None,  default=None, 
        help=("If given, then it will start training from this ckpt"
              "It has higher prioty than official_ckpt_name, but lower than the ckpt found in autoresuming (see trainer.py) "
              "It must be given if inpaint_mode is true")
    )
    
    parser.add_argument('--inpaint_mode', default=False, type=lambda x:x.lower() == "true", help="Train a GLIGEN model in inpaitning setting")
    parser.add_argument('--randomize_fg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance that fg mask will not be a box but a random mask. See code for details")
    parser.add_argument('--random_add_bg_mask', default=False, type=lambda x:x.lower() == "true", help="Only used if inpaint_mode is true. If true, 0.5 chance add arbitrary mask for the whole image. See code for details")
    
    parser.add_argument('--enable_ema', default=False, type=lambda x:x.lower() == "true")
    parser.add_argument("--ema_rate", type=float,  default=0.9999, help="")
    parser.add_argument("--total_iters", type=int,  default=500000, help="")
    parser.add_argument("--save_every_iters", type=int,  default=5000, help="")
    parser.add_argument("--disable_inference_in_training", type=lambda x:x.lower() == "true",  default=False, help="Do not do inference, thus it is faster to run first a few iters. It may be useful for debugging ")

    args = parser.parse_args()
    assert args.scheduler_type in ['cosine', 'constant']
    

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    config = OmegaConf.load(args.yaml_file) 
    config.update( vars(args) )
    config.total_batch_size = config.batch_size * n_gpu
    if args.inpaint_mode:
        config.model.params.inpaint_mode = True


    trainer = Trainer_Autoencoder(config)
    synchronize()
    trainer.start_training()

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_autoencoder4.py  --yaml_file=configs/mcxface.yaml  --DATA_ROOT=./DATA/mcxface/   --batch_size=1   --save_every_iters 1000   --name mcxface