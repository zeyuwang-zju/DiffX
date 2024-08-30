from tkinter.messagebox import NO
import torch 
import json 
from collections import defaultdict
from PIL import Image, ImageDraw
from copy import deepcopy
import os 
import torchvision.transforms as transforms
import torchvision
from .base_dataset import BaseDataset, check_filenames_in_zipdata, recalculate_box_and_verify_if_valid 
from io import BytesIO
import random

from io import BytesIO
import base64
from PIL import Image
import numpy as np

from ldm.util import instantiate_from_config


def mask_for_random_drop_text_or_image_feature(masks, random_drop_embedding):
    """
    input masks tell how many valid grounding tokens for this image
    e.g., 1,1,1,1,0,0,0,0,0,0...

    If random_drop_embedding=both.  we will random drop either image or
    text feature for each token, 
    but we always make sure there is at least one feature used. 
    In other words, the following masks are not valid 
    (because for the second obj, no feature at all):
    image: 1,0,1,1,0,0,0,0,0
    text:  1,0,0,0,0,0,0,0,0

    if random_drop_embedding=image. we will random drop image feature 
    and always keep the text one.  

    """
    N = masks.shape[0]

    if random_drop_embedding=='both':
        temp_mask = torch.ones(2,N)
        for i in range(N):
            if random.uniform(0, 1) < 0.5: # else keep both features 
                idx = random.sample([0,1], 1)[0] # randomly choose to drop image or text feature 
                temp_mask[idx,i] = 0 
        image_masks = temp_mask[0]*masks
        text_masks = temp_mask[1]*masks
    
    if random_drop_embedding=='image':
        image_masks = masks*(torch.rand(N)>0.5)*1
        text_masks = masks

    return image_masks, text_masks



class FLIRDataset(BaseDataset):
    def __init__(self, 
                flir_path,
                prob_use_caption=1,
                random_drop_embedding='none',
                image_size=512, 
                min_box_size=0.0001,
                max_boxes_per_data=8,
                max_images=None,
                random_crop = True,
                random_flip = True,
                train_or_test = 'train',
                ):
        super().__init__(random_crop, random_flip, image_size)
        self.flir_path = flir_path
        self.prob_use_caption = prob_use_caption
        self.random_drop_embedding = random_drop_embedding
        self.min_box_size = min_box_size
        self.max_boxes_per_data = max_boxes_per_data
        self.max_images = max_images

        embed_person, embed_bicycle, embed_car = self.get_text_embeddings()

        self.class_anno = {0: embed_person, 1: embed_bicycle, 2: embed_car}

        if self.max_boxes_per_data > 99:
            assert False, "Are you sure setting such large number of boxes per image?"

        self.embedding_len = 768

        file = os.path.join(flir_path, 'Main', 'train.txt') if train_or_test == 'train' else os.path.join(flir_path, 'Main', 'test.txt')
        with open(file, 'r') as file:
            self.file_list = [line.strip() for line in file.readlines()]


    def total_images(self):
        return len(self.file_list)


    def get_text_embeddings(self):
        os.makedirs(os.path.join(self.flir_path, 'text_embeddings'), exist_ok=True)

        embed_person = torch.load(os.path.join(self.flir_path, 'text_embeddings/person.pt'))
        embed_bicycle = torch.load(os.path.join(self.flir_path, 'text_embeddings/bicycle.pt'))
        embed_car = torch.load(os.path.join(self.flir_path, 'text_embeddings/car.pt'))

        return embed_person, embed_bicycle, embed_car



    def __getitem__(self, index):
        out = {}
        item = self.file_list[index]

        # -------------------- id and image ------------------- # 
        out['id'] = index
        out['item'] = item
        image_RGB_path = os.path.join(self.flir_path, 'JPEGImages_RGB', f'{item}.jpg')
        image_TIR_path = os.path.join(self.flir_path, 'JPEGImages_TIR', f'{item}.jpg')
        image_RGB = Image.open(image_RGB_path).convert('RGB')
        image_TIR = Image.open(image_TIR_path).convert('RGB')
        image_tensor_RGB, image_tensor_TIR, trans_info, flip = self.transform_image_pair(image_RGB, image_TIR)
        out["image_RGB"] = image_tensor_RGB.detach()
        out["image_TIR"] = image_tensor_TIR.detach()

        # -------------------- grounding token ------------------- # 
        label_path = os.path.join(self.flir_path, 'labels', f'{item}.txt')

        areas = []
        all_boxes = []
        all_masks = []
        all_text_embeddings = []

        all_labels = []

        with open(label_path, 'r') as file:
            for line in file:
                label_info = line.strip().split()
                
                object_class = int(label_info[0])
                x_center = float(label_info[1])
                y_center = float(label_info[2])
                width = float(label_info[3])
                height = float(label_info[4])

                x = (x_center - width / 2) * 640 
                y = (y_center - height / 2) * 512
                w = width * 640
                h = height * 512

                valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(x, y, w, h, trans_info, self.image_size, self.min_box_size)
                if valid:
                    areas.append(  (x1-x0)*(y1-y0)  )
                    all_boxes.append( torch.tensor([x0,y0,x1,y1]) / self.image_size ) # scale to 0-1
                    all_masks.append(1)
                    all_text_embeddings.append(self.class_anno[object_class])
                    all_labels.append(object_class)

        # Sort according to area and choose the largest N objects   
        wanted_idxs = torch.tensor(areas).sort(descending=True)[1]
        wanted_idxs = wanted_idxs[0:self.max_boxes_per_data]

        boxes = torch.zeros(self.max_boxes_per_data, 4)
        masks = torch.zeros(self.max_boxes_per_data)
        text_embeddings =  torch.zeros(self.max_boxes_per_data, self.embedding_len)
        labels = torch.zeros(self.max_boxes_per_data)

        for i, idx in enumerate(wanted_idxs):
            boxes[i] = all_boxes[idx]
            masks[i] = all_masks[idx]
            text_embeddings[i] =  all_text_embeddings[idx]
            labels[i] = all_labels[idx]

        if self.random_drop_embedding != 'none':
            image_masks, text_masks = mask_for_random_drop_text_or_image_feature(masks, self.random_drop_embedding)
        else:
            image_masks = masks
            text_masks = masks

        out["boxes"] = boxes.detach()
        out["masks"] = masks.detach() # indicating how many valid objects for this image-text data
        out["image_masks"] = image_masks.detach() # indicating how many objects still there after random dropping applied
        out["text_masks"] = text_masks.detach() # indicating how many objects still there after random dropping applied
        out["text_embeddings"] =  text_embeddings.detach()
        out["image_embeddings"] =  torch.zeros_like(text_embeddings, requires_grad=False)  
        out["labels"] =  labels.detach()

        # -------------------- caption ------------------- # 
        if random.uniform(0, 1) < self.prob_use_caption:
            caption_path = os.path.join(self.flir_path, 'caption', f'{item}.txt')
            with open(caption_path, "r") as caption:
                out["caption"] = caption.read()
            if flip == True:
                out["caption"] = out["caption"].replace("the right", "temp").replace("the left", "the right").replace("temp", "the left")
        else:
            out["caption"] = ""

        return out

    def __len__(self):
        return len(self.file_list)