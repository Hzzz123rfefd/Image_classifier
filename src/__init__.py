from src.model import ModelUnetForImageClassifier
from src.dataset import DatasetForImageClassifier


datasets = {
   "image_classifier": DatasetForImageClassifier
}

models = {
    "image_classifier_base_unet": ModelUnetForImageClassifier,
}