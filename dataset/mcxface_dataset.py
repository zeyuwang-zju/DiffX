from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw, ImageOps
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid 
from io import BytesIO
import random
import math

from io import BytesIO
import base64
from PIL import Image
import numpy as np

from ldm.util import instantiate_from_config


class MCXFaceDataset(BaseDataset):
    def __init__(self, 
                mcxface_path,
                prob_use_caption=1,
                image_size=512, 
                random_crop = True,
                random_flip = True,
                train_or_test = 'train',
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.mcxface_path = mcxface_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()

        file = os.path.join(mcxface_path, 'Main', 'train.txt') if train_or_test == 'train' else os.path.join(mcxface_path, 'Main', 'test.txt')
        with open(file, 'r') as file:
            self.file_list = [line.strip() for line in file.readlines()]


    def total_images(self):
        return len(self.file_list)


    def __getitem__(self, index):
        out = {}
        item = self.file_list[index]

        # -------------------- id and image ------------------- # 
        out['id'] = index
        out['item'] = item
        image_VIS_path = os.path.join(self.mcxface_path, 'JPEGImages_VIS', f'{item}.jpg')
        image_NIR_path = os.path.join(self.mcxface_path, 'JPEGImages_NIR', f'{item}.jpg')
        image_SWIR_path = os.path.join(self.mcxface_path, 'JPEGImages_SWIR', f'{item}.jpg')
        image_THERMAL_path = os.path.join(self.mcxface_path, 'JPEGImages_THERMAL', f'{item}.jpg')
        image_3DDFA_path = os.path.join(self.mcxface_path, 'JPEGImages_3DDFA', f'{item}.jpg')

        image_VIS = Image.open(image_VIS_path).convert('RGB')
        image_NIR = Image.open(image_NIR_path).convert('RGB')
        image_SWIR = Image.open(image_SWIR_path).convert("RGB")
        image_THERMAL = Image.open(image_THERMAL_path).convert('RGB')
        image_3DDFA = Image.open(image_3DDFA_path).convert('RGB')

        # - - - - - center_crop, resize and random_flip - - - - - - #

        if self.random_crop:
            width, height = image_VIS.size
            scale_factor = max(self.image_size / width, self.image_size / height) * random.uniform(1 , 1 / 0.95)
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            image_VIS = image_VIS.resize((new_width, new_height), Image.NEAREST)
            image_NIR = image_NIR.resize((new_width, new_height), Image.NEAREST)
            image_SWIR = image_SWIR.resize((new_width, new_height), Image.NEAREST)
            image_THERMAL = image_THERMAL.resize((new_width, new_height), Image.NEAREST)
            image_3DDFA = image_3DDFA.resize((new_width, new_height), Image.NEAREST)

            crop_x = random.randint(0, new_width - self.image_size)
            crop_y = random.randint(0, new_height - self.image_size)
            
            image_VIS = image_VIS.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_NIR = image_NIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SWIR = image_SWIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_THERMAL = image_THERMAL.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_3DDFA = image_3DDFA.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))

        else:
            width, height = image_VIS.size
            scale_factor = max(self.image_size / width, self.image_size / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            image_VIS = image_VIS.resize((new_width, new_height), Image.NEAREST)
            image_NIR = image_NIR.resize((new_width, new_height), Image.NEAREST)
            image_SWIR = image_SWIR.resize((new_width, new_height), Image.NEAREST)
            image_THERMAL = image_THERMAL.resize((new_width, new_height), Image.NEAREST)
            image_3DDFA = image_3DDFA.resize((new_width, new_height), Image.NEAREST)

            crop_x = (new_width - self.image_size) // 2
            crop_y = (new_height - self.image_size) // 2

            image_VIS = image_VIS.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_NIR = image_NIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SWIR = image_SWIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_THERMAL = image_THERMAL.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_3DDFA = image_3DDFA.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))


        flip = False
        if self.random_flip and random.random()<0.5:
            flip = True
            image_VIS = ImageOps.mirror(image_VIS)
            image_NIR = ImageOps.mirror(image_NIR)
            image_SWIR = ImageOps.mirror(image_SWIR)    
            image_THERMAL = ImageOps.mirror(image_THERMAL)    
            image_3DDFA = ImageOps.mirror(image_3DDFA)    

        out["image_VIS"] = ( self.pil_to_tensor(image_VIS).float()/255 - 0.5 ) / 0.5
        out["image_NIR"] = ( self.pil_to_tensor(image_NIR).float()/255 - 0.5 ) / 0.5 
        out["image_SWIR"] = ( self.pil_to_tensor(image_SWIR).float()/255 - 0.5 ) / 0.5
        out["image_THERMAL"] = ( self.pil_to_tensor(image_THERMAL).float()/255 - 0.5 ) / 0.5 
        out["image_3DDFA"] = ( self.pil_to_tensor(image_3DDFA).float()/255 - 0.5 ) / 0.5 
        out["sem"] = out["image_3DDFA"]
        out['mask'] = torch.tensor(1.0) 

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            caption_path = os.path.join(self.mcxface_path, 'caption', f'{item}.txt')
            with open(caption_path, "r") as caption:
                out["caption"] = caption.read()
            if flip == True:
                out["caption"] = out["caption"].replace("the right", "temp").replace("the left", "the right").replace("temp", "the left")
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.file_list)