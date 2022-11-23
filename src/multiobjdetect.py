# Il nostro codice
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle as pk
from torchvision.datasets import ImageFolder

from google.colab import drive
drive.mount('/content/drive')

# Caricamento dei dati
if not os.path.exists("assignment_1"):
    ! unzip -P dluniud2022 /content/drive/Shareddrives/Razor/Root/4Anno/DeepLearning/datasets/cldataset.zip 

#train and test data directory
data_dir = "/content/assignment_1/train/images"
test_data_dir = "/content/assignment_1/test/images"

#load the train and test data
dataset = ImageFolder()


# Carica il tensor
import torch
from torchvision import transforms
from PIL import Image

img = Image.open("/content/assignment_1/train/images/152832.jpg")
img
convert_tensor = transforms.ToTensor()
convert_tensor(img)