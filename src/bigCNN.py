# -*- coding: utf-8 -*-
#Untitled13 (1).ipyn

#!unzip -P dluniud2022 ./drive/MyDrive/dataset.zip

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
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import time
from PIL import Image
from sklearn.model_selection import train_test_split

scale = 112 
samples = 10000
active_train = True
num_of_epochs = 500
image_dir = './assignment_1/train/'
load_model = False
number_model = 501
lr = 0.0000075

def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    
    return ret


def preprocess_dataset():
    # Lists that will contain the whole dataset
    labels = []
    boxes = []
    images = []

    rows = os.listdir(image_dir + "images")[:samples]
    cicle = 0
    for row in rows[:samples]:
        stringa = str(row[0:6]) + ".json"
        with open(image_dir + "annotations/" + stringa, 'r') as f:
            data = json.loads(f.read())
            array = items(data)
            img_path = row
            image = cv2.imread(os.path.join(image_dir + "images/",img_path))
            h = image.shape[0] / scale
            w = image.shape[1] / scale
            image = cv2.resize(image, (scale, scale))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / (255.0) 
            if True: #array[0]['category_name'] != "short sleeve top":
              boxes.append([])
              labels.append([])
              for item in array:
                  lista = item['bounding_box']
                  lista[0] = lista[0]/scale/w
                  lista[1] = lista[1]/scale/h
                  lista[2] = lista[2]/scale/w
                  lista[3] = lista[3]/scale/h
                  boxes[cicle].append(item['bounding_box'])
                  labels[cicle].append(item['category_name'])
              images.append(image)
              cicle += 1
            #boxes.append(lista)          #da salvare smontato
            #labels.append(array[0]['category_name'])

            
    return labels, boxes, images

def display_boxes(labels, boxes, img_list):
    combined_list = list(zip(img_list, boxes, labels))
    # Generate a random sample of images each time the cell is run 
    random_range = random.sample(range(1, len(img_list)), 1)

    for itr, i in enumerate(random_range, 1):
        img_size = scale 
        image  = img_list[i] 

        # Draw bounding boxes on the image
        for box in boxes[i]:
          a1, b1, a2, b2 = box
          x1 = a1 * scale
          x2 = a2 * scale
          y1 = b1 * scale
          y2 = b2 * scale
          cv2.rectangle(image, (int(x1),int(y1)),
            (int(x2),int(y2)),
                    (255,0,0),
                    1)
        #x1 = a1 * width[i]
        #x2 = a2 * height[i]
        #y1 = b1 * height[i]
        #y2 = b2 * width[i]
        print(a1, x1, b1, y1)
        print(a2, x2, b2, y2)
        #print()
        #cv2.rectangle(image, (int(x1),int(y1)), (int(x2),int(y2)),  (255,0,0), 1)
    
        # Clip the values to 0-1 and draw the sample of images
        img = np.clip(image, 0, 1)
        plt.imshow(image)
    plt.show()


def change(label):
  if label == 'long sleeve dress':
      return 0
  elif label == 'long sleeve outwear':
      return 1
  elif label == 'long sleeve top':
      return 2
  elif label == 'short sleeve dress':
      return 3
  elif label == 'short sleeve outwear':
      return 4
  elif label == 'short sleeve top':
      return 5
  elif label == 'shorts':
      return 6
  elif label == 'skirt':
      return 7
  elif label == 'sling':
      return 8
  elif label == 'sling dress':
      return 9
  elif label == 'trousers':
      return 10
  elif label == 'vest':
      return 11
  elif label == 'vest dress':
      return 12
  else:
      return -1

# Load images, labels, boxes
labels, boxes, img_list = preprocess_dataset()
#print(boxes)
#display_boxes(labels, boxes, img_list)

#train_images = np.array(img_list)
#train_boxes = np.array(boxes)

classes = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 
'short sleeve dress', 'short sleeve outwear', 'short sleeve top', 'short', 'skirt', 'sling', 'sling dress', 'trousers', 'vest', 'vest dress']

train_labels = []
train_boxes = []
#print(labels)

for label in labels:
  train_labels.append([change(label[i]) if i<len(label) else -1 for i in range(max(6, len(label)))])

for box in boxes:
  train_boxes.append([box[i] if i<len(box) else [-1, -1, -1, -1] for i in range(max(6, len(box)))])
#train_labels = np.array(train_labels)

train_images, val_images, train_labels, \
val_labels, train_boxes, val_boxes = train_test_split( np.array(img_list), 
                np.array(train_labels), np.array(train_boxes), test_size = 0.2, 
                random_state = 43)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

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

dataset = Dataset(train_images, train_labels, train_boxes)
valdataset = ValDataset(val_images, val_labels, val_boxes)
print("Creato dataset")

def initialize_weights(m):
  if isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # CNN
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=12, stride=2, kernel_size=7, padding = 4), nn.BatchNorm2d(12))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=12, out_channels= 25, kernel_size=3, padding = 3), nn.BatchNorm2d(25))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=25, out_channels= 75, kernel_size=3, padding = 2), nn.BatchNorm2d(75))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=75, out_channels= 150, kernel_size=3, padding = 2), nn.BatchNorm2d(150))

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=150, out_channels= 300, kernel_size=3, padding = 2), nn.BatchNorm2d(300))
        
        self.conv6 = nn.Conv2d(in_channels=300, out_channels= 300, kernel_size=3, padding = 1)

        self.conv7 = nn.Conv2d(in_channels=300, out_channels= 512, kernel_size=3)

        #self.conv8 = nn.Conv2d(in_channels=2048, out_channels= 2048, kernel_size=3)

        self.class_fc1 = nn.Linear(in_features=512, out_features=300)
        self.class_fc1_2 = nn.Linear(in_features=300, out_features=150)
        self.class_fc2 = nn.Linear(in_features=150, out_features=20)
        self.class_out = nn.Linear(in_features=20, out_features=13)
		
        self.class_fc2a = nn.Linear(in_features=150, out_features=20)
        self.class_outa = nn.Linear(in_features=20, out_features=13)
        
        self.box_fc1 = nn.Linear(in_features=512, out_features=300)
        self.box_fc1_2 = nn.Linear(in_features=300, out_features=150)
        self.box_fc2 = nn.Linear(in_features=150, out_features=20)
        self.box_out = nn.Linear(in_features=20, out_features=4)
		
        self.box_outa = nn.Linear(in_features=20, out_features=4)


        self.relu = nn.LeakyReLU(inplace = True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(0.4)

    def forward(self, t):
        t = self.conv1(t)
        t = self.relu(t)
        t = self.max_pool(t)
        
        t = self.conv2(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv3(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv4(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv5(t)
        t = self.relu(t)
        t = self.max_pool(t)

        t = self.conv6(t) 

        t = self.conv7(t) 
        #t = self.conv8(t) 

        t = torch.flatten(t, start_dim=1)

        #print(len(t))
        class_t = self.class_fc1(t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t = self.class_fc1_2(class_t)
        class_t = self.relu(class_t)
        class_t = self.drop(class_t)

        class_t1 = self.class_fc2(class_t)
        class_t1 = self.relu(class_t1)
        class_t1 = self.drop(class_t1)
		
        class_t1 = F.softmax(self.class_out(class_t1), dim=1)


        class_t2 = self.class_fc2a(class_t)
        class_t2 = self.relu(class_t2)
        class_t2 = self.drop(class_t2)

        class_t2 = F.softmax(self.class_outa(class_t2), dim=1)


        box_t = self.box_fc1(t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)

        box_t = self.box_fc1_2(box_t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)

        box_t = self.box_fc2(box_t)
        box_t = self.relu(box_t)
        box_t = self.drop(box_t)

        box_t0 = self.box_out(box_t)
        box_t0 = F.sigmoid(box_t0)

        box_t1 = self.box_outa(box_t)
        box_t1 = F.sigmoid(box_t1)

        return [class_t1, box_t0, class_t2, box_t1]

print("Creata cnn")
model = Network()
model.apply(initialize_weights)
if load_model:
    model.load_state_dict(torch.load("models/model_ep"+str(number_model)+".pth"))
model = model.to(device)


model

def get_num_correct(preds, labels):
  #print(preds)
  #.eq(labels).sum().item())
  return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=True)

def train(model):
    # Defining the optimizer
    optimizer = optim.Adam(model.parameters(), lr)
    epochs = []
    losses = []
    # Creating a directory for storing models
    if not os.path.isdir('models'):
        os.mkdir('models')
    for epoch in range(num_of_epochs):
        tot_loss = 0
        tot_correct = 0
        train_start = time.time()
        model.train()
        for batch, (x, y, z) in enumerate(dataloader):
        	# Converting data from cpu to GPU if available to improve speed
            x,y1,z1 = x.to(device),y[:,0].to(device),z[:,0].to(device)
            
            optimizer.zero_grad()
            [y_pred, z_pred, y1_pred, z1_pred]= model(x)
            class_loss = 0
            box_loss = 0

            class_loss = F.cross_entropy(y_pred, y1)
            box_loss = F.mse_loss(z_pred, z1)

            (class_loss+box_loss).backward(retain_graph=True)

            y2,z2 = y[:,1].to(device),z[:,1].to(device)
            y2 = torch.where(y2 < 0, y1, y2)
            z2 = torch.where(z2 < 0, z1, z2)
            class_loss = F.cross_entropy(y1_pred, y2)
            box_loss = F.mse_loss(z1_pred, z2)
            optimizer.zero_grad()
            (class_loss+box_loss).backward()

            optimizer.step()

            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-train_start)/60, end='\r')
        model.eval()
        for batch, (x, y, z) in enumerate(valdataloader):
        	
            x,y,z = x.to(device),y[:,0].to(device),z[:,0].to(device)
            
            optimizer.zero_grad()
            with torch.no_grad():
                [y_pred,z_pred, y1_pred, z1_pred]= model(x)
                
                class_loss = F.cross_entropy(y_pred, y)
                box_loss = F.mse_loss(z_pred, z)
            tot_loss += (class_loss.item() + box_loss.item())
            #print(class_loss.item())
            #print(box_loss.item())
            tot_correct += get_num_correct(y_pred, y)
            print("Test batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-train_start)/60, end='\r')
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch", epoch, "Accuracy", (tot_correct)/(samples / 32), "loss:",
              tot_loss/(samples / 32), " time: ", (time.time()-train_start)/60, " mins")
        if(epoch%50 == 0):
          torch.save(model.state_dict(), "models/model_ep"+str(epoch+1)+".pth")

print("Creato train")   
#print(device) 
if active_train:
    train(model)
torch.save(model.state_dict(), "models/model_ep"+str(1000)+".pth")
print("Eseguito train")

def preprocess(img, image_size = scale):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image

def postprocess(image, results):
    [class_probs, bounding_box, class_probs1, bounding_box1] = results
    class_index = torch.argmax(class_probs)
    class_label = classes[class_index]

    class_index1 = torch.argmax(class_probs1)
    class_label1 = classes[class_index1]
    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]
    x3, y3, x4, y4 = bounding_box1[0]
    print(results)
    # # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(scale * x1)
    x2 = int(scale * x2)
    y1 = int(scale * y1)
    y2 = int(scale * y2)

    x3 = int(scale * x3)
    x4 = int(scale * x4)
    y3 = int(scale * y3)
    y4 = int(scale * y4)
    # return the lable and coordinates

    return class_label, (x1,y1,x2,y2), torch.max(class_probs)*100, class_label1, (x3,y3,x4,y4), torch.max(class_probs1)*100,

def predict(image):
    model = Network()
    model = model.to(device)
    model.load_state_dict(torch.load("models/model_ep"+str(number_model)+".pth"))
    model.eval()
    
    # Reading Image
    img  = cv2.imread(image)
    h = img.shape[0] / scale
    w = img.shape[1] / scale
    # # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(img)
    result = model(torch.permute(torch.from_numpy(processed_image).float(),(0,3,1,2)).to(device))

    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence, label1, (x3, y3, x4, y4), confidence1 = postprocess(image, result)

    img = cv2.resize(img, (scale, scale))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 1)
    cv2.rectangle(img, (x3, y3), (x4, y4), (255, 0, 100), 1)

    #cv2.putText(img, '{}, CONFIDENCE: {}'.format(label, confidence), (5, int(15 * 0.5)), cv2.FONT_HERSHEY_COMPLEX, 0.15, (200, 55, 100), 1)
    #cv2.putText(img, '{}, CONFIDENCE: {}'.format(label1, confidence1), (5, int(20 * 0.5)), cv2.FONT_HERSHEY_COMPLEX, 0.15, (200, 55, 100), 1)

    print("Informazioni")
    print(label, confidence.item())
    print(label1, confidence1.item())

    plt.imshow(img[:,:,::-1])
    plt.show()

import gc
gc.collect()
model = ""
torch.cuda.empty_cache()

while(True):
    imcode = input("Codice: ")
    image = "./assignment_1/test/images/"+imcode+".jpg"
    predict(image)