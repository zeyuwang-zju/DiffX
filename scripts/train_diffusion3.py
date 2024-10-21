import os, sys
sys.path.append(os.getcwd())
import torch
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
import torchvision
from convert_ckpt import add_additional_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from copy import deepcopy
from inpaint_mask_func import draw_masks_from_boxes
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #


class ImageCaptionSaver:
    def __init__(self, base_path, nrow=4, normalize=True, scale_each=True, range=(-1,1) ):
        self.base_path = base_path 
        self.nrow = nrow
        self.normalize = normalize
        self.scale_each = scale_each
        self.range = range

    def __call__(self, images, real, masked_real, sem, captions, seen):
        
        save_path = os.path.join(self.base_path, str(seen).zfill(8)+'.png')
        if sem is None:
            torchvision.utils.save_image( torch.cat([real, images], dim=0), save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )
        else:
            torchvision.utils.save_image( torch.cat([sem, real, images], dim=0), save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range )

        if masked_real is not None:
            # only inpaiting mode case 
            save_path = os.path.join(self.base_path, str(seen).zfill(8)+'_mased_real.png')
            torchvision.utils.save_image( masked_real, save_path, nrow=self.nrow, normalize=self.normalize, scale_each=self.scale_each, range=self.range)

        assert images.shape[0] == len(captions) * 3

        save_path = os.path.join(self.base_path, 'captions.txt')
        with open(save_path, "a") as f:
            f.write( str(seen).zfill(8) + ':\n' )    
            for cap in captions:
                if cap == "":
                    f.write( "No caption." + '\n' ) 
                else:
                    f.write( cap + '\n' )  
            f.write( '\n' ) 



def read_official_ckpt(ckpt_path):      
    "Read offical pretrained SD ckpt and convert into my style" 
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    out = {}
    out["model"] = {}
    out["text_encoder"] = {}
    out["autoencoder"] = {}
    out["unexpected"] = {}
    out["diffusion"] = {}

    for k,v in state_dict.items():
        if k.startswith('model.diffusion_model'):
            out["model"][k.replace("model.diffusion_model.", "")] = v 
        elif k.startswith('cond_stage_model'):
            out["text_encoder"][k.replace("cond_stage_model.", "")] = v 
        elif k.startswith('first_stage_model'):
            out["autoencoder"][k.replace("first_stage_model.", "")] = v 
        elif k in ["model_ema.decay", "model_ema.num_updates"]:
            out["unexpected"][k] = v  
        else:
            out["diffusion"][k] = v     
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
        for batch in loader:  # TODO: it seems each time you have the same order for all epoch?? 
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
    checkpoint_autoencoder = None

    if os.path.exists(name):
        all_tags = os.listdir(name)
        all_existing_tags = [ tag for tag in all_tags if tag.startswith('tag')    ]
        all_existing_tags.sort()
        all_existing_tags = all_existing_tags[::-1]
        for previous_tag in all_existing_tags:
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
                break 

        for previous_tag in all_existing_tags:
            potential_ckpt_autoencoder = os.path.join( name, previous_tag, 'autoencoder_checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt_autoencoder):
                checkpoint_autoencoder = potential_ckpt_autoencoder
                if get_rank() == 0:
                    print('auto-resuming autoencoder ckpt found '+ potential_ckpt_autoencoder)
                break 

        curr_tag = 'tag'+str(len(all_existing_tags)).zfill(2)
        name = os.path.join( name, curr_tag ) # output/name/tagxx
    else:
        name = os.path.join( name, 'tag00' ) # output/name/tag00

    if get_rank() == 0:
        os.makedirs(name) 
        os.makedirs(  os.path.join(name,'Log')  ) 
        writer = SummaryWriter( os.path.join(name,'Log')  )

    return name, writer, checkpoint, checkpoint_autoencoder


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 



class Trainer:
    def __init__(self, config):

        self.config = config
        self.device = torch.device("cuda")

        self.l_simple_weight = 1
        self.name, self.writer, checkpoint, checkpoint_autoencoder = create_expt_folder_with_auto_resuming(config.OUTPUT_ROOT, config.name)
        if get_rank() == 0:
            shutil.copyfile(config.yaml_file, os.path.join(self.name, "train_config_file.yaml")  )
            self.config_dict = vars(config)
            torch.save(  self.config_dict,  os.path.join(self.name, "config_dict.pth")     )

        # = = = = = = = = = = = = = = = = = create model and diffusion = = = = = = = = = = = = = = = = = #
        self.model = instantiate_from_config(config.model).to(self.device)
        self.autoencoder = instantiate_from_config(config.autoencoder).to(self.device)
        self.text_encoder = instantiate_from_config(config.text_encoder).to(self.device)
        self.diffusion = instantiate_from_config(config.diffusion).to(self.device)

        
        state_dict = read_official_ckpt(  os.path.join(config.DATA_ROOT, config.official_ckpt_name)   )
        
        # modify the input conv for SD if necessary (grounding as unet input; inpaint)
        additional_channels = self.model.additional_channel_from_downsampler
        if self.config.inpaint_mode:
            additional_channels += 5 # 5 = 4(latent) + 1(mask)
        add_additional_channels(state_dict["model"], additional_channels)
        self.input_conv_train = True if additional_channels>0 else False

        # load original SD ckpt (with inuput conv may be modified) 
        missing_keys, unexpected_keys = self.model.load_state_dict( state_dict["model"], strict=False  )
        assert unexpected_keys == []
        original_params_names = list( state_dict["model"].keys()  ) # used for sanity check later 
        self.diffusion.load_state_dict( state_dict["diffusion"]  )
 
        self.autoencoder.eval()
        self.text_encoder.eval()
        disable_grads(self.autoencoder)
        disable_grads(self.text_encoder)

        # = = = = = = = = = = = = = load from ckpt: (usually for inpainting training) = = = = = = = = = = = = = #
        if self.config.ckpt is not None:
            first_stage_ckpt = torch.load(self.config.ckpt, map_location="cpu")
            self.model.load_state_dict(first_stage_ckpt["model"])


        # = = = = = = = = = = = = = = = = = create opt = = = = = = = = = = = = = = = = = #
        params = []
        trainable_names = []
        all_params_name = []
        for name, p in self.model.named_parameters():
            # print(name)
            if ("transformer_blocks" in name) and ("fuser" in name):
                # New added Attention layers 
                params.append(p) 
                trainable_names.append(name)
            elif  "position_net" in name:
                # Grounding token processing network 
                params.append(p) 
                trainable_names.append(name)
            elif  "downsample_net" in name:
                # Grounding downsample network (used in input) 
                params.append(p) 
                trainable_names.append(name)
            elif (self.input_conv_train) and ("input_blocks.0.0.weight" in name):
                # First conv layer was modified, thus need to train 
                params.append(p) 
                trainable_names.append(name)
            elif "cross_objs_context" in name:
                params.append(p)
                trainable_names.append(name)
            else:
                # Following make sure we do not miss any new params
                # all new added trainable params have to be haddled above
                # otherwise it will trigger the following error  
                assert name in original_params_names, name 
            all_params_name.append(name) 


        self.opt = torch.optim.AdamW(params, lr=config.base_learning_rate, weight_decay=config.weight_decay) 
        count_params(params)

        #  = = = = = EMA... It is worse than normal model in early experiments, thus never enabled later = = = = = = = = = #
        if config.enable_ema:
            self.master_params = list(self.model.parameters()) 
            self.ema = deepcopy(self.model)
            self.ema_params = list(self.ema.parameters())
            self.ema.eval()

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
        sampler_train = DistributedSampler(dataset_train, seed=config.seed) if config.distributed else None 
        loader_train = DataLoader( dataset_train,  batch_size=config.batch_size, 
                                                   shuffle=(sampler_train is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler_train)
        self.dataset_train = dataset_train
        self.loader_train = wrap_loader(loader_train)

        dataset_val = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=False, repeats=train_dataset_repeats)
        sampler_val = DistributedSampler(dataset_val, seed=config.seed) if config.distributed else None 
        loader_val = DataLoader( dataset_val,  batch_size=config.batch_size, 
                                                   shuffle=(sampler_val is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler_val)
        self.dataset_val = dataset_val
        self.loader_val = wrap_loader(loader_val)


        if get_rank() == 0:
            total_image = dataset_train.total_images()
            print("Total training images: ", total_image)  

            total_image_val = dataset_val.total_images()
            print("Total Validation images: ", total_image_val)        
        

        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            print(f"Loading autoencoder and diffusion from {checkpoint}......")
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if config.enable_ema:
                self.ema.load_state_dict(checkpoint["ema"])
            self.opt.load_state_dict(checkpoint["opt"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.starting_iter = checkpoint["iters"]
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])
            if self.starting_iter >= config.total_iters:
                synchronize()
                print("Training finished. Start exiting")
                exit()
        
        else:
            print(f"Loading autoencoder from {checkpoint_autoencoder}, while diffusion if initialized......")
            checkpoint_autoencoder = torch.load(checkpoint_autoencoder, map_location="cpu")
            self.autoencoder.load_state_dict(checkpoint_autoencoder["autoencoder"])

        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if get_rank() == 0:       
            self.image_caption_saver = ImageCaptionSaver(self.name, nrow=self.config.batch_size * 3)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )


    @torch.no_grad()
    def get_input(self, batch):

        z = self.autoencoder.encode( batch["image_RGB"], batch["image_D"], batch["image_TIR"] )
        context = self.text_encoder.encode( batch["caption"]  )

        _t = torch.rand(z.shape[0]).to(z.device)
        t = (torch.pow(_t, 1) * 1000).long()
        t = torch.where(t!=1000, t, 999) # if 1000, then replace it with 999
        
        inpainting_extra_input = None
        if self.config.inpaint_mode:
            # extra input for the inpainting model 
            inpainting_mask = draw_masks_from_boxes(batch['boxes'], 64, randomize_fg_mask=self.config.randomize_fg_mask, random_add_bg_mask=self.config.random_add_bg_mask).cuda()
            masked_z = z*inpainting_mask
            inpainting_extra_input = torch.cat([masked_z,inpainting_mask], dim=1)              
        
        grounding_extra_input = None
        if self.grounding_downsampler_input != None:
            grounding_extra_input = self.grounding_downsampler_input.prepare(batch)

        return z, t, context, inpainting_extra_input, grounding_extra_input 


    def run_one_step(self, batch):
        x_start, t, context, inpainting_extra_input, grounding_extra_input = self.get_input(batch)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion.q_sample(x_start=x_start, t=t, noise=noise)

        grounding_input = self.grounding_tokenizer_input.prepare(batch)
        input = dict(x=x_noisy, 
                    timesteps=t, 
                    context=context, 
                    inpainting_extra_input=inpainting_extra_input,
                    grounding_extra_input=grounding_extra_input,
                    grounding_input=grounding_input)
        model_output = self.model(input)
        
        loss = torch.nn.functional.mse_loss(model_output, noise) * self.l_simple_weight

        self.loss_dict = {"loss": loss.item()}

        return loss 
        

    def start_training(self):

        iterator = tqdm(range(self.starting_iter, self.config.total_iters), desc='Training progress',  disable=get_rank() != 0 )
        self.model.train()
        for iter_idx in iterator: # note: iter_idx is not from 0 if resume training
            self.iter_idx = iter_idx

            self.opt.zero_grad()
            batch = next(self.loader_train)
            batch_to_device(batch, self.device)

            loss = self.run_one_step(batch)
            loss.backward()
            self.opt.step() 
            self.scheduler.step()
            if self.config.enable_ema:
                update_ema(self.ema_params, self.master_params, self.config.ema_rate)


            if (get_rank() == 0):
                if (iter_idx % 10 == 0):
                    self.log_loss() 
                if (iter_idx == 0)  or  ( iter_idx % self.config.save_every_iters == 0 )  or  (iter_idx == self.config.total_iters-1):
                    self.save_ckpt_and_result()
            synchronize()
        
        synchronize()
        print("Training finished. Start exiting")
        exit()


    def log_loss(self):
        for k, v in self.loss_dict.items():
            self.writer.add_scalar(  k, v, self.iter_idx+1  )  # we add 1 as the actual name
    

    @torch.no_grad()
    def save_ckpt_and_result(self):

        model_wo_wrapper = self.model.module if self.config.distributed else self.model

        iter_name = self.iter_idx + 1     # we add 1 as the actual name

        if not self.config.disable_inference_in_training:
            batch_here = self.config.batch_size
            batch = sub_batch( next(self.loader_val), batch_here)
            batch_to_device(batch, self.device)

            batch_here = batch["image_RGB"].shape[0]

            if "boxes" in batch:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    temp_data = {"image": batch["image_RGB"][i], "boxes":batch["boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)

                    temp_data = {"image": batch["image_TIR"][i], "boxes":batch["boxes"][i]}
                    im = self.dataset_train.datasets[0].vis_getitem_data(out=temp_data, return_tensor=True, print_caption=False)
                    real_images_with_box_drawing.append(im)

                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            else:
                real_images_with_box_drawing = [] # we save this durining trianing for better visualization
                for i in range(batch_here):
                    real_images_with_box_drawing.append(batch["image_RGB"][i]*0.5 + 0.5)
                    real_images_with_box_drawing.append(batch["image_D"][i]*0.5 + 0.5)
                    real_images_with_box_drawing.append(batch["image_TIR"][i]*0.5 + 0.5)

                real_images_with_box_drawing = torch.stack(real_images_with_box_drawing)
            
            uc = self.text_encoder.encode( batch_here*[""] )
            context = self.text_encoder.encode(  batch["caption"]  )
            
            plms_sampler = PLMSSampler(self.diffusion, model_wo_wrapper)      
            shape = (batch_here, model_wo_wrapper.in_channels, model_wo_wrapper.image_size, model_wo_wrapper.image_size)
            
            inpainting_extra_input = None
            grounding_extra_input = None
            if self.grounding_downsampler_input != None:
                grounding_extra_input = self.grounding_downsampler_input.prepare(batch)
            
            grounding_input = self.grounding_tokenizer_input.prepare(batch)
            input = dict( x=None, 
                        timesteps=None, 
                        context=context, 
                        inpainting_extra_input=inpainting_extra_input,
                        grounding_extra_input=grounding_extra_input,
                        grounding_input=grounding_input )
            samples = plms_sampler.sample(S=50, shape=shape, input=input, uc=uc, guidance_scale=5)
            
            autoencoder_wo_wrapper = self.autoencoder # Note itself is without wrapper since we do not train that. 
            dec_RGB, dec_D, dec_TIR = autoencoder_wo_wrapper.decode(samples)
            dec_RGB = torch.clamp(dec_RGB, min=-1, max=1).cpu()
            dec_D = torch.clamp(dec_D, min=-1, max=1).cpu()
            dec_TIR = torch.clamp(dec_TIR, min=-1, max=1).cpu()

            samples = []
            for i in range(batch_here):
                samples.append(dec_RGB[i])
                samples.append(dec_D[i])
                samples.append(dec_TIR[i])
            samples = torch.stack(samples)
            masked_real_image = None

            if "sem" in batch:
                if batch["sem"].shape[1] == 9:
                    color_map = {
                        0: [0, 0, 0],
                        1: [255, 0, 0],
                        2: [0, 255, 0],
                        3: [0, 0, 255],
                        4: [255, 255, 0],
                        5: [255, 0, 255],
                        6: [0, 255, 255],
                        7: [128, 0, 0],
                        8: [0, 128, 0]
                    }
                    num_classes = 9

                elif batch["sem"].shape[1] == 2:
                    color_map = {
                        0: [0, 0, 0],
                        1: [255, 255, 255],
                    }
                    num_classes = 2
                
                else:
                    color_map = batch["sem"]
                    num_classes = None


                if num_classes == None:
                    save_sem = []
                    for i in range(batch_here):
                        save_sem.append(batch["sem"][i])
                        save_sem.append(batch["sem"][i])
                    save_sem = torch.stack(save_sem).cpu()
                    self.image_caption_saver(samples.cpu(), real_images_with_box_drawing.cpu(),  masked_real_image, save_sem, batch["caption"], iter_name)

                else:
                    save_sem = []
                    for i in range(batch_here):
                        sem = batch["sem"][i]
                        h, w = sem.shape[1:]
                        color_image = np.zeros((h, w, 3), dtype=np.uint8)

                        for class_index in range(num_classes):
                            color = color_map[class_index]
                            mask = sem[class_index] == 1
                            color_image[mask.cpu()] = color

                        torch_image = torch.from_numpy(color_image.transpose((2, 0, 1))).float() / 255
                        save_sem.append(torch_image)
                        save_sem.append(torch_image)
                        save_sem.append(torch_image)

                    save_sem = torch.stack(save_sem).cpu()

                    self.image_caption_saver(samples.cpu(), real_images_with_box_drawing.cpu(),  masked_real_image, save_sem, batch["caption"], iter_name)

            else:
                self.image_caption_saver(samples, real_images_with_box_drawing,  masked_real_image, None, batch["caption"], iter_name)


        ckpt = dict(model = model_wo_wrapper.state_dict(),
                    autoencoder = self.autoencoder.state_dict(),
                    diffusion = self.diffusion.state_dict(),
                    opt = self.opt.state_dict(),
                    scheduler= self.scheduler.state_dict(),
                    iters = self.iter_idx+1,
                    config_dict=self.config_dict,
        )
        if self.config.enable_ema:
            ckpt["ema"] = self.ema.state_dict()
        torch.save( ckpt, os.path.join(self.name, "checkpoint_"+str(iter_name).zfill(8)+".pth") )
        torch.save( ckpt, os.path.join(self.name, "checkpoint_latest.pth") )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from distributed import synchronize
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


    trainer = Trainer(config)
    synchronize()
    trainer.start_training()

    # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_diffusion3.py  --yaml_file=configs/come_sobel_sod.yaml  --DATA_ROOT=./DATA/come/   --batch_size=8   --save_every_iters 1000   --name come3

    # CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch --nproc_per_node=2 scripts/train_diffusion3.py  --yaml_file=configs/mfnet_triple.yaml  --DATA_ROOT=/data16_2/wangzeyu/dataset/mfnet/  --OUTPUT_ROOT /data16_2/wangzeyu/diffx_output/   --batch_size=8   --save_every_iters 1000   --name mfnet3