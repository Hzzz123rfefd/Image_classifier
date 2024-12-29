import math
import cv2
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from torch import optim
from torch.utils.data import DataLoader

from src.modules import *
from src.utils import *


class Unet(nn.Module):
    def __init__(self, in_c = 3, out_c = 1024):
        super(Unet, self).__init__()
        # 4次下采样: double_conv + max_pool
        self.conv1 = DoubleConv(in_c, 64)
        self.pool1 = DownSample(64, 64)  # 1/2

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = DownSample(128, 128)  # 1/4

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = DownSample(256, 256)  # 1/8

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = DownSample(512, 512)  # 1/16
        self.conv5 = DoubleConv(512, out_c)

    def forward(self, x):
        # 4次下采样: double_conv + max_pool
        c1 = self.conv1(x)
        p1 = self.pool1(c1)  # 1/2
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)  # 1/16
        c5 = self.conv5(p4)
        return c5
    
    
class ModelUnetForImageClassifier(nn.Module):
    def __init__(
        self, 
        target_width,
        target_height,
        in_channel = 3,
        hidden_channel = 1024,
        num_class = 2,
        device = "cpu" #      "cuda"
    ):
        super().__init__()
        self.target_width = target_width
        self.target_height = target_height
        self.num_class = num_class
        self.device = device if torch.cuda.is_available() else "cpu"
        self.unet = Unet(in_c = in_channel, out_c = hidden_channel)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层 全局平均池化 -> (b, c, 1, 1)
        self.classifier = nn.Linear(hidden_channel, self.num_class) 

    def forward(self,input: dict):
        images = input["image"].to(self.device)
        labels = input["label"].to(self.device)
        x = self.unet(images)
        x = self.global_avg_pool(x)
        predict = self.classifier(torch.flatten(x, 1))   # (b, c)
        classs = torch.argmax(predict,dim = -1)
        output = {
            "predict":predict, # (b, 10) 
            "label":labels  #(b)   
        }
        return output

    def compute_loss(self, input: dict):
        output = {}
        if "class_weights" not in input:
            input["class_weights"] = None
        if "mask" not in input:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"])
            output["total_loss"] = self.criterion(input["predict"],input["label"])
        else:
            self.criterion = nn.CrossEntropyLoss(weight = input["class_weights"], reduction = 'none')
            loss = self.criterion(input["predict"],input["label"])
            masked_loss = loss * input["mask"]
            output["total_loss"] = masked_loss.sum() / input["mask"].sum()
        return output

    def trainning(
        self,
        train_dataloader:DataLoader = None,
        test_dataloader:DataLoader = None,
        optimizer_name:str = "Adam",
        weight_decay: float = 1e-4,
        clip_max_norm: float = 0.5,
        factor: float = 0.3,
        patience: int = 15,
        lr:float = 1e-4,
        total_epoch:int = 1000,
        save_checkpoint_step: str = 10,
        save_model_dir:str = "models"
    ):
        ## 1 trainning log path 
        first_trainning = True
        check_point_path = save_model_dir  + "/checkpoint.pth"
        log_path = save_model_dir + "/train.log"

        ## 2 get net pretrain parameters if need 
        """
            If there is training history record, load pretrain parameters
        """
        if  os.path.isdir(save_model_dir) and os.path.exists(check_point_path) and os.path.exists(log_path):
            self.load_pretrained(save_model_dir)  
            first_trainning = False

        else:
            if not os.path.isdir(save_model_dir):
                os.makedirs(save_model_dir)
            with open(log_path, "w") as file:
                pass


        ##  3 get optimizer
        if optimizer_name == "Adam":
            optimizer = optim.Adam(params = self.parameters(), lr = lr,  weight_decay = weight_decay)
        elif optimizer_name == "AdamW":
            optimizer = optim.AdamW(params = self.parameters(), lr = lr,  weight_decay = weight_decay)
        else:
            optimizer = optim.Adam(params = self.parameters(), lr = lr,  weight_decay = weight_decay)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer, 
            mode = "min", 
            factor = factor, 
            patience = patience
        )

        ## init trainng log
        if first_trainning:
            best_loss = float("inf")
            last_epoch = 0
        else:
            checkpoint = torch.load(check_point_path, map_location=self.device)
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_loss = checkpoint["loss"]
            last_epoch = checkpoint["epoch"] + 1

        try:
            for epoch in range(last_epoch, total_epoch):
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_loss = self.train_one_epoch(epoch, train_dataloader, optimizer, clip_max_norm, log_path)
                test_loss = self.test_epoch(epoch, test_dataloader, log_path)
                loss = train_loss + test_loss
                lr_scheduler.step(loss)
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                check_point_path = save_model_dir  + "/checkpoint.pth"
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": None,
                        "lr_scheduler": None
                    },
                    check_point_path
                )

                if epoch % save_checkpoint_step == 0:
                    os.makedirs(save_model_dir + "/" + "chaeckpoint-"+str(epoch))
                    torch.save(
                        {
                            "epoch": epoch,
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict()
                        },
                        save_model_dir + "/" + "chaeckpoint-"+str(epoch)+"/checkpoint.pth"
                    )
                if is_best:
                    self.save_pretrained(save_model_dir)

        # interrupt trianning
        except KeyboardInterrupt:
                torch.save(                
                    {
                        "epoch": epoch,
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict()
                    },
                    check_point_path
                )

    def train_one_epoch(self, epoch, train_dataloader, optimizer, clip_max_norm, log_path = None):
        self.train()
        self.to(self.device)
        pbar = tqdm(train_dataloader,desc="Processing epoch "+str(epoch), unit="batch")
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        for batch_id, inputs in enumerate(train_dataloader):
            """ grad zeroing """
            optimizer.zero_grad()

            """ forward """
            output = self.forward(inputs)

            """ calculate loss """
            out_criterion = self.compute_loss(output)
            out_criterion["total_loss"].backward()
            total_loss.update(out_criterion["total_loss"].item())
            average_hit_rate.update(math.exp(-total_loss.avg))

            """ grad clip """
            if clip_max_norm > 0:
                clip_gradient(optimizer,clip_max_norm)

            """ modify parameters """
            optimizer.step()
            postfix_str = "total_loss: {:.4f},average_hit_rate:{:.4f}".format(
                total_loss.avg,
                average_hit_rate.avg
            )
            pbar.set_postfix_str(postfix_str)
            pbar.update()
        with open(log_path, "a") as file:
            file.write(postfix_str+"\n")
        return total_loss.avg

    def test_epoch(self, epoch, test_dataloader,trainning_log_path = None):
        total_loss = AverageMeter()
        average_hit_rate = AverageMeter()
        self.eval()
        self.to(self.device)
        with torch.no_grad():
            for batch_id, inputs in enumerate(test_dataloader):
                """ forward """
                output = self.forward(inputs)

                """ calculate loss """
                out_criterion = self.compute_loss(output)
                total_loss.update(out_criterion["total_loss"].item())

            average_hit_rate.update(math.exp(-total_loss.avg))
            postfix_str = "total_loss: {:.4f}, average_hit_rate:{:.4f}".format(
                total_loss.avg, 
                average_hit_rate.avg
            )
        print(postfix_str)
        with open(trainning_log_path, "a") as file:
            file.write(postfix_str + "\n")
        return total_loss.avg

    def load_pretrained(self, save_model_dir):
        self.load_state_dict(torch.load(save_model_dir + "/model.pth"))

    def save_pretrained(self,  save_model_dir):
        torch.save(self.state_dict(), save_model_dir + "/model.pth")
        
    def inference(self, image_path):
        self.eval()
        image_data = []
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)    # (h,w,c)  numpy -> (b,c,h,w) tensor
        if len(resized_image.shape) == 2:  
            resized_image = np.expand_dims(image, axis=0)
        image_data.append(resized_image)
        dataset = np.transpose(np.array(image_data), (0, 3, 1, 2))
        dataset = dataset / 255.0
        image = torch.tensor(dataset[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            x = self.unet(image)
            x = self.global_avg_pool(x)
            predict = self.classifier(torch.flatten(x, 1))   # (b, c)
        classs = torch.argmax(predict,dim = -1)    
        return classs.cpu().item()
        