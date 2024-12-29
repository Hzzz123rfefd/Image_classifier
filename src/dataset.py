import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from src.utils import *

class DatasetForImageReader(Dataset):
    def __init__(
        self, 
        target_width: int,
        target_height: int,
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        data_type:str = "train"
    ):
        self.data_type = data_type
        self.target_width = target_width
        self.target_height = target_height
        
        if data_type == "train":
            self.use_image_folder = train_image_folder
        elif data_type == "test":
            self.use_image_folder = test_image_folder
        elif data_type == "valid":
            self.use_image_folder = valid_image_folder

        image_data = []
        files = os.listdir(self.use_image_folder)
        files = [f for f in files if os.path.isfile(os.path.join(self.use_image_folder, f))]
        files = sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))

        for filename in files:
            img_file_path = os.path.join(self.use_image_folder, filename)
            image_data.append(self.read_image(img_file_path))
        
        self.dataset = np.transpose(np.array(image_data), (0, 3, 1, 2))  # np.array(image_data)    (n,h,w,c)   ->  (n,c,h,w) 
        
        self.dataset = self.dataset / 255.0
        self.total_samples = len(self.dataset)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        return output
    
    def collate_fn(self,batch):
        return recursive_collate_fn(batch)
    
    def read_image(self, image_path):
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        if len(resized_image.shape) == 2:  
            resized_image = np.expand_dims(image, axis=0)
        return resized_image


class DatasetForImageClassifier(DatasetForImageReader):
    def __init__( 
        self, 
        target_width:int,
        target_height:int,
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        train_annotation_file:str = None,
        test_annotation_file:str = None,
        valid_annotation_file:str = None,
        data_type:str = "train"
    ):
        super().__init__(target_width, target_height, train_image_folder, test_image_folder, valid_image_folder, data_type)
        if self.data_type == "train":
            self.use_annotation_file = train_annotation_file
        elif self.data_type == "test":
            self.use_annotation_file = test_annotation_file
        elif self.data_type == "valid":
            self.use_annotation_file = valid_annotation_file
        
        with open(self.use_annotation_file, 'r') as f:
            annotations = json.load(f)

        label = [annotation["label"] for annotation in annotations]
        self.label = np.array(label)

    def __getitem__(self, idx):
        output = {}
        output["image"] = torch.tensor(self.dataset[idx], dtype=torch.float32)         # 
        output["label"] = torch.tensor(self.label[idx], dtype=torch.int64)                 #
        return output
    
    def collate_fn(self, batch):
        return recursive_collate_fn(batch)
        