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

image_dir = './assignment_1/train/'

def preprocess_dataset():
    # Lists that will contain the whole dataset
    labels = []
    boxes = []
    images = []
    scale = 256

    rows = os.listdir(image_dir + "images")[::]
    cicle = 0
    for row in rows[:21]:
        stringa = str(row[0:6]) + ".json"
        with open(image_dir + "annotations/" + stringa, 'r') as f:
            data = json.loads(f.read())
            array = items(data)
            
            img_path = row
            image = cv2.imread(os.path.join(image_dir + "images/",img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / (255.0) 

            boxes.append([])
            labels.append([])
            images.append(image)
            for item in array:
                boxes[cicle].append(item['bounding_box'])
                labels[cicle].append(item['category_name'])

            cicle += 1

    return labels, boxes, images 

def display_boxes(labels, boxex, img_list):
    combined_list = list(zip(img_list, boxes, labels))
    # Generate a random sample of images each time the cell is run 
    random_range = random.sample(range(1, len(img_list)), 20)

    for itr, i in enumerate(random_range, 1):
        img_size = 256 
        image  = img_list[i] 

        # Draw bounding boxes on the image
        for box in boxes[i]:
            a1, b1, a2, b2 = box
            x1 = a1
            x2 = a2
            y1 = b1
            y2 = b2
            print(a1, x1, b1, y1)
            print(a2, x2, b2, y2)
            print()
            cv2.rectangle(image, (int(x1),int(y1)),
                (int(x2),int(y2)),
                        (255,0,0),
                        3)
    
        # Clip the values to 0-1 and draw the sample of images
        #img = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()

# Load images, labels, boxes
labels, boxes, img_list = preprocess_dataset()

display_boxes(labels, boxes, img_list)



