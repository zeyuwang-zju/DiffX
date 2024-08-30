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


class MFNetTripleDataset(BaseDataset):
    def __init__(self, 
                mfnet_path,
                prob_use_caption=1,
                image_size=512, 
                random_crop = True,
                random_flip = True,
                train_or_test = 'train',
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.mfnet_path = mfnet_path
        self.prob_use_caption = prob_use_caption
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.pil_to_tensor = transforms.PILToTensor()

        file = os.path.join(mfnet_path, 'Main', 'trainval.txt') if train_or_test == 'train' else os.path.join(mfnet_path, 'Main', 'test.txt')
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
        image_RGB_path = os.path.join(self.mfnet_path, 'JPEGImages_RGB', f'{item}.jpg')
        image_TIR_path = os.path.join(self.mfnet_path, 'JPEGImages_TIR', f'{item}.jpg')
        image_SOD_path = os.path.join(self.mfnet_path, 'JPEGImages_SOD', f'{item}.jpg')
        sem_path = os.path.join(self.mfnet_path, 'labels', f'{item}.png')

        image_RGB = Image.open(image_RGB_path).convert('RGB')
        image_TIR = Image.open(image_TIR_path).convert('RGB')
        image_SOD = Image.open(image_SOD_path).convert("L") # semantic class index 0,1,2,3,4 in uint8 representation 
        image_SEM = Image.open(sem_path).convert('RGB') # semantic class index 0,1,2,3,4 in uint8 representation 

        # - - - - - center_crop, resize and random_flip - - - - - - #

        if self.random_crop:
            width, height = image_RGB.size
            scale_factor = max(self.image_size / width, self.image_size / height) * random.uniform(1 , 1 / 0.8)
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image_RGB = image_RGB.resize((new_width, new_height), Image.NEAREST)
            image_TIR = image_TIR.resize((new_width, new_height), Image.NEAREST)
            image_SOD = image_SOD.resize((new_width, new_height), Image.NEAREST)
            image_SEM = image_SEM.resize((new_width, new_height), Image.NEAREST)

            crop_x = random.randint(0, new_width - self.image_size)
            crop_y = random.randint(0, new_height - self.image_size)
            
            image_RGB = image_RGB.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_TIR = image_TIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SOD = image_SOD.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SEM = image_SEM.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))

        else:
            width, height = image_RGB.size
            scale_factor = max(self.image_size / width, self.image_size / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image_RGB = image_RGB.resize((new_width, new_height), Image.NEAREST)
            image_TIR = image_TIR.resize((new_width, new_height), Image.NEAREST)
            image_SOD = image_SOD.resize((new_width, new_height), Image.NEAREST)
            image_SEM = image_SEM.resize((new_width, new_height), Image.NEAREST)

            crop_x = (new_width - self.image_size) // 2
            crop_y = (new_height - self.image_size) // 2

            image_RGB = image_RGB.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_TIR = image_TIR.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SOD = image_SOD.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))
            image_SEM = image_SEM.crop((crop_x, crop_y, crop_x+self.image_size, crop_y+self.image_size))

        flip = False
        if self.random_flip and random.random()<0.5:
            flip = True
            image_RGB = ImageOps.mirror(image_RGB)
            image_TIR = ImageOps.mirror(image_TIR)
            image_SOD = ImageOps.mirror(image_SOD)
            image_SEM = ImageOps.mirror(image_SEM)    

        image_SOD = self.pil_to_tensor(image_SOD)[0,:,:] // 255

        input_label = torch.zeros(2, self.image_size, self.image_size)
        image_SOD = input_label.scatter_(0, image_SOD.long().unsqueeze(0), 1.0)

        out["image_RGB"] = ( self.pil_to_tensor(image_RGB).float()/255 - 0.5 ) / 0.5
        out["image_D"] = ( self.pil_to_tensor(image_TIR).float()/255 - 0.5 ) / 0.5 
        out["image_sobel"] = ( self.pil_to_tensor(image_SEM).float()/255 - 0.5 ) / 0.5
        out['sem'] = image_SOD
        out['mask'] = torch.tensor(1.0) 

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            caption_path = os.path.join(self.mfnet_path, 'caption', f'{item}.txt')
            with open(caption_path, "r") as caption:
                out["caption"] = caption.read()
            if flip == True:
                out["caption"] = out["caption"].replace("the right", "temp").replace("the left", "the right").replace("temp", "the left")
        else:
            out["caption"] = ""

        return out


    def __len__(self):
        return len(self.file_list)