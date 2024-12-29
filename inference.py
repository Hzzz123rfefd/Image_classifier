import argparse
import json
import os

from tqdm import tqdm
from src import models
from utils import *

def main(args):
    config = load_config(args.model_config_path)
    """ get net struction"""
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])
    print("label:", net.inference(args.image_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/image_classifier.yml")
    parser.add_argument("--image_path",type=str,default = "./mnist_train\\images/train\\0000002.png")
    args = parser.parse_args()
    main(args)