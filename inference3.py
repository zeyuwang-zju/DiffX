import torch
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
import numpy as np
from dataset.concat_dataset import ConCatDataset #, collate_fn
from torch.utils.data.distributed import  DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os 
import shutil
import torchvision
from convert_ckpt import add_additional_channels
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from distributed import get_rank, synchronize
# = = = = = = = = = = = = = = = = = = useful functions = = = = = = = = = = = = = = = = = #


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
        for batch in loader:
            yield batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


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
            potential_ckpt = os.path.join( name, previous_tag, 'checkpoint_latest.pth' )
            if os.path.exists(potential_ckpt):
                checkpoint = potential_ckpt
                if get_rank() == 0:
                    print('auto-resuming ckpt found '+ potential_ckpt)
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


class Inference:
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


        # # = = = = = = = = = = = = = = = = = = = = create data = = = = = = = = = = = = = = = = = = = = #  
        train_dataset_repeats = config.train_dataset_repeats if 'train_dataset_repeats' in config else None  # None
        dataset_val = ConCatDataset(config.train_dataset_names, config.DATA_ROOT, train=False, repeats=train_dataset_repeats)
        sampler_val = DistributedSampler(dataset_val, seed=config.seed) if config.distributed else None 
        loader_val = DataLoader( dataset_val,  batch_size=config.batch_size, 
                                                   shuffle=(sampler_val is None),
                                                   num_workers=config.workers, 
                                                   pin_memory=True, 
                                                   sampler=sampler_val)
        self.dataset_val = dataset_val
        self.loader_val = wrap_loader(loader_val)


        # = = = = = = = = = = = = = = = = = = = = load from autoresuming ckpt = = = = = = = = = = = = = = = = = = = = #
        self.starting_iter = 0  
        if checkpoint is not None:
            print(f"Loading autoencoder and diffusion from {checkpoint}......")
            checkpoint = torch.load(checkpoint, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.autoencoder.load_state_dict(checkpoint["autoencoder"])
        else:
            print(f"No pretrained checkpoint.")
            exit()

        # = = = = = = = = = = = = = = = = = = = = misc and ddp = = = = = = = = = = = = = = = = = = = =#    
        # func return input for grounding tokenizer 
        self.grounding_tokenizer_input = instantiate_from_config(config.grounding_tokenizer_input)
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        
        # func return input for grounding downsampler  
        self.grounding_downsampler_input = None
        if 'grounding_downsampler_input' in config:
            self.grounding_downsampler_input = instantiate_from_config(config.grounding_downsampler_input)

        if config.distributed:
            self.model = DDP( self.model, device_ids=[config.local_rank], output_device=config.local_rank, broadcast_buffers=False )


    def start_inference(self):
        
        os.makedirs(os.path.join(self.name, 'dec_RGB'), exist_ok=True) 
        os.makedirs(os.path.join(self.name, 'dec_D'), exist_ok=True)
        os.makedirs(os.path.join(self.name, 'dec_Sobel'), exist_ok=True)
        self.model.eval()

        model_wo_wrapper = self.model.module if self.config.distributed else self.model
        for batch in tqdm(self.loader_val): # note: iter_idx is not from 0 if resume training
            batch_to_device(batch, self.device)
            batch_here = batch["image_RGB"].shape[0]
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
            dec_RGB, dec_D, dec_Sobel = autoencoder_wo_wrapper.decode(samples)
            dec_RGB = torch.clamp(dec_RGB, min=-1, max=1).cpu()
            dec_D = torch.clamp(dec_D, min=-1, max=1).cpu()
            dec_Sobel = torch.clamp(dec_Sobel, min=-1, max=1).cpu()

            item = batch['item'][0]
            torchvision.utils.save_image( dec_RGB, os.path.join(self.name, 'dec_RGB', f'{item}.jpg'), nrow=1, normalize=True, scale_each=True, range=(-1,1))
            torchvision.utils.save_image( dec_D, os.path.join(self.name, 'dec_D', f'{item}.jpg'), nrow=1, normalize=True, scale_each=True, range=(-1,1))
            torchvision.utils.save_image( dec_Sobel, os.path.join(self.name, 'dec_Sobel', f'{item}.jpg'), nrow=1, normalize=True, scale_each=True, range=(-1,1))

            if "sem" in batch:
                os.makedirs(os.path.join(self.name, 'sem'), exist_ok=True)
                if batch["sem"].shape[2] == 9:
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
                else:
                    color_map = {
                        0: [0, 0, 0],
                        1: [255, 255, 255],
                    }
                    num_classes = 2

                sem = batch["sem"][0]
                h, w = sem.shape[1:]
                color_image = np.zeros((h, w, 3), dtype=np.uint8)

                for class_index in range(num_classes):
                    color = color_map[class_index]
                    mask = sem[class_index] == 1
                    color_image[mask.cpu()] = color

                torch_image = torch.from_numpy(color_image.transpose((2, 0, 1))).float() / 255
                torchvision.utils.save_image( torch_image, os.path.join(self.name, 'sem', f'{item}.jpg'), nrow=1, normalize=True, scale_each=True, range=(-1,1))

        synchronize()
        print("Inference finished. Start exiting")
        exit()


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
    parser.add_argument("--local_rank", type=int, default=1)
    parser.add_argument("--yaml_file", type=str,  default="configs/flickr.yaml", help="paths to base configs.")

    parser.add_argument("--base_learning_rate", type=float,  default=5e-5, help="")
    parser.add_argument("--weight_decay", type=float,  default=0.0, help="")
    parser.add_argument("--warmup_steps", type=int,  default=10000, help="")
    parser.add_argument("--scheduler_type", type=str,  default='constant', help="cosine or constant")
    parser.add_argument("--batch_size", type=int,  default=1, help="")
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


    config.train_dataset_names.SODSobelGrounding.prob_use_caption = 1
    config.train_dataset_names.SODSobelGrounding.random_crop = False
    config.train_dataset_names.SODSobelGrounding.random_flip = False

    trainer = Inference(config)
    synchronize()
    trainer.start_inference()

    # CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 inference3.py  --yaml_file=configs/come_sobel_sod.yaml  --DATA_ROOT=./DATA/come/  --name come3

