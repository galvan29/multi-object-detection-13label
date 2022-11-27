import os
import cv2
import json
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
import random
import torch
from torchvision import transforms
from PIL import Image

def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    
    return ret

image_dir = '/home/fdsa/Documents/assignment_1/train/'

def preprocess_dataset():
    # Lists that will contain the whole dataset
    labels = []
    boxes = []

    h = 256
    w = 256

    rows = os.listdir(image_dir + "images")[:21]
    cicle = 0
    for row in rows[:21]:
        stringa = str(row[0:6]) + ".json"
        with open(image_dir + "annotations/" + stringa, 'r') as f:
            data = json.loads(f.read())
            array = items(data)
            
            #img_path = row
            #img  = cv2.imread(os.path.join(image_dir + "images/",img_path))
            #image = cv2.resize(img, (64, 64))
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = image.astype("float") / 255.0

            
            boxes.append([])
            labels.append([])

            for item in array:
                boxes[cicle].append(item['bounding_box'])
                labels[cicle].append(item['category_name'])

            cicle += 1

    return labels, boxes, rows 

labels, boxes, img_list = preprocess_dataset()
combined_list = list(zip(img_list, boxes, labels))


# Create a Matplotlib figure
#plt.figure(figsize=(20,20))

# Generate a random sample of images each time the cell is run 
random_range = random.sample(range(1, len(img_list)), 20)

for itr, i in enumerate(random_range, 1):
    # Bounding box of each image
    
    img_size = 256
    ## Rescaling the boundig box values to match the image size
    
    # The image to visualize
    img  = cv2.imread(os.path.join(image_dir + "images/", img_list[i]))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / (img_size-1)

    # Draw bounding boxes on the image
    for box in boxes[i]:
        a1, b1, a2, b2 = box
        cv2.rectangle(image, (int(a1),int(b1)),
            (int(a2),int(b2)),
                    (255,0,0),
                    3)
    
    # Clip the values to 0-1 and draw the sample of images
    img = np.clip(image, 0, 1)
    plt.subplot(4, 5, itr)
    plt.imshow(img)
    plt.axis('off')

plt.show()