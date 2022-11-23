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
import json
import shutil

from google.colab import drive
drive.mount('/content/drive')

# Caricamento dei dati
if not os.path.exists("assignment_1"):
    ! unzip -P dluniud2022 /content/drive/Shareddrives/Razor/Root/4Anno/DeepLearning/datasets/cldataset.zip 

#train and test data directory
data_dir = "/content/assignment_1/train/images"
test_data_dir = "/content/assignment_1/test/images"

#load the train and test data
#dataset = ImageFolder()


# Carica il tensor
import torch
from torchvision import transforms
from PIL import Image



#gaveiro
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
import json
import shutil

from google.colab import drive
drive.mount('/content/drive')

# Caricamento dei dati
if not os.path.exists("assignment_1"):
    ! unzip -P dluniud2022 /content/drive/Shareddrives/Razor/Root/4Anno/DeepLearning/datasets/cldataset.zip 

#train and test data directory
data_dir = "/content/assignment_1/train/images"
test_data_dir = "/content/assignment_1/test/images"


# Carica il tensor
import torch
from torchvision import transforms
from PIL import Image



#gaveiro

#shapes of img
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                     (0.5, 0.5, 0.5))])


#create folder of class and insert img
path = "/content/assignment_1/train/images/"
dir_list = os.listdir(path)
if not os.path.exists('/content/assignment_1/train/class/'):
  os.mkdir('/content/assignment_1/train/class/')

for name in dir_list:
  stringa = str(name[0:6]) + ".json"
  with open('/content/assignment_1/train/annotations/' + stringa, 'r') as f:
    data = json.load(f)
    dirName = '/content/assignment_1/train/class/' + data['item1']['category_name']
    if not os.path.exists(dirName):
      os.mkdir(dirName)
    if not os.path.exists(dirName + name):
      shutil.copy('/content/assignment_1/train/images/' + name, dirName)
   
   
   
#create Tensor
path = "/content/assignment_1/train/class/"

dataset = ImageFolder(path, transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))



#load tensor in train
trainloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                          shuffle=True, num_workers=2)


# show some img of training
path = "/content/assignment_1/train/class/"
classes = os.listdir(path)

dataiter = iter(trainloader)
images, labels = dataiter.next()

for img, label in zip(images, labels):
    img_unnorm = img/2 + 0.5
    img_transp = np.transpose(img_unnorm, (1,2,0))

    fig = plt.figure(figsize=(2,2))
    plt.imshow(img_transp)
    plt.title(classes[label.item()], fontweight="bold")

    plt.show()
