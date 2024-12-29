
import argparse
import json
import os

from tqdm import tqdm
from src import models
from utils import *

datasets = []
with open("mnist_train/annotations/test/test_annotations.json", 'r') as f:
    annotations = json.load(f)
label = [annotation["label"] for annotation in annotations]

datasets = []
for root, _, files in os.walk("mnist_train/images/test"):
    for file in files:
        if file.endswith('.png'):
            datasets.append(os.path.join(root, file))

            
def calculate_accuracy(labels, predictions):
    assert len(labels) == len(predictions), "Labels and predictions must have the same length"
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    accuracy = correct / len(labels)
    print("accuracy:", accuracy)
    return accuracy

def main(args):
    config = load_config(args.model_config_path)
    """ get net struction"""
    net = models[config["model_type"]](**config["model"])
    net.load_pretrained(config["logging"]["save_dir"])
    predicts = []
    for image_path in tqdm(datasets, desc="Processing data"):
        predicts.append(net.inference(image_path))

    calculate_accuracy(label, predicts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path",type=str,default = "config/image_classifier.yml")
    args = parser.parse_args()
    main(args)