import os
import cv2
import json
import torch
from sklearn.model_selection import train_test_split
import numpy as np

def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    return ret

def process_dataset(image_dir = "./assignment_1/train/", samples = 200, scale = 160, active_train = False, changes={}):
    if not active_train: return
    
    # ret values
    labels  = []
    boxes   = []
    images  = []

    # loading
    rows = os.listdir(image_dir + "images")[:samples]
    cicle = 0

    for row in rows[:samples]:
        imageinfo_filename = str(row[0:6]) + ".json"

        with open(image_dir + "annotations/" + imageinfo_filename, 'r') as f:
            data = json.loads(f.read())
            array = items(data)

            # loading and scaling 
            img_path = row
            image = cv2.imread(os.path.join(image_dir + "images/",img_path))
            h = image.shape[0] / scale
            w = image.shape[1] / scale
            image = cv2.resize(image, (scale, scale))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / (255.0) 
            
            if True: 
              boxes.append([])
              labels.append([])
              for item in array:
                  lista = item['bounding_box']

                  lista[0] = lista[0]/scale/w
                  lista[1] = lista[1]/scale/h
                  lista[2] = lista[2]/scale/w
                  lista[3] = lista[3]/scale/h

                  boxes[cicle].append(lista)
                  labels[cicle].append(item['category_name'])
              images.append(image)
              cicle += 1
    
    train_labels = []
    train_boxes = []
    #print(labels)

    for label in labels:
        train_labels.append([changes[label[i]] if i<len(label) else -1 for i in range(max(6, len(label)))])

    for box in boxes:
        train_boxes.append([box[i] if i<len(box) else [-1, -1, -1, -1] for i in range(max(6, len(box)))])

    train_images, val_images, train_labels, \
    val_labels, train_boxes, val_boxes = train_test_split(
        np.array(images), 
        np.array(train_labels), np.array(train_boxes), test_size = 0.2, 
        random_state = 43)

    return labels, boxes, images, train_images, val_images, train_labels, val_labels, train_boxes, val_boxes


class Dataset():
    def __init__(self, train_images, train_labels, train_boxes):
        self.images = torch.permute(torch.from_numpy(train_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(train_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(train_boxes).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.images[idx],
              self.labels[idx],
              self.boxes[idx])

class ValDataset(Dataset):

    def __init__(self, val_images, val_labels, val_boxes):
        self.images = torch.permute(torch.from_numpy(val_images),(0,3,1,2)).float()
        self.labels = torch.from_numpy(val_labels).type(torch.LongTensor)
        self.boxes = torch.from_numpy(val_boxes).float()