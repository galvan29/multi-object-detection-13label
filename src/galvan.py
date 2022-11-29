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


def items(jsstring):
    ret = []
    for i in range(1,20):
        if "item"+str(i) in jsstring:
            ret.append(jsstring["item"+str(i)])
    
    return ret

image_dir = '../assignment_1/train/'

def preprocess_dataset():
    # Lists that will contain the whole dataset
    labels = []
    boxes = []
    images = []
    scale = 256

    rows = os.listdir(image_dir + "images")[::]
    cicle = 0
    for row in rows[:1000]:
        stringa = str(row[0:6]) + ".json"
        with open(image_dir + "annotations/" + stringa, 'r') as f:
            data = json.loads(f.read())
            array = items(data)
            img_path = row
            image = cv2.imread(os.path.join(image_dir + "images/",img_path))
            h = image.shape[0] / 256
            w = image.shape[1] / 256
            image = cv2.resize(image, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / (255.0) 
            #boxes.append([])
            #labels.append([])
            images.append(image)
            #for item in array:
                #boxes[cicle].append(item['bounding_box'])
                #labels[cicle].append(item['category_name'])
            lista = array[0]['bounding_box']
            lista[0] = lista[0]/256/w
            lista[1] = lista[1]/256/h
            lista[2] = lista[2]/256/w
            lista[3] = lista[3]/256/h
            boxes.append(lista)          #da salvare smontato
            labels.append(array[0]['category_name'])

            cicle += 1
    return labels, boxes, images

def display_boxes(labels, boxes, img_list):
    combined_list = list(zip(img_list, boxes, labels))
    # Generate a random sample of images each time the cell is run 
    random_range = random.sample(range(1, len(img_list)), 1)

    for itr, i in enumerate(random_range, 1):
        img_size = 256 
        image  = img_list[i] 

        # Draw bounding boxes on the image
        #for box in boxes[i]:
        a1, b1, a2, b2 = boxes[i]
        x1 = a1 * 256
        x2 = a2 * 256
        y1 = b1 * 256
        y2 = b2 * 256

        #x1 = a1 * width[i]
        #x2 = a2 * height[i]
        #y1 = b1 * height[i]
        #y2 = b2 * width[i]
        print(a1, x1, b1, y1)
        print(a2, x2, b2, y2)
        #print()
        cv2.rectangle(image, (int(x1),int(y1)),
            (int(x2),int(y2)),
                    (255,0,0),
                    3)
    
        # Clip the values to 0-1 and draw the sample of images
        img = np.clip(image, 0, 1)
        plt.imshow(image)
    plt.show()

# Load images, labels, boxes
labels, boxes, img_list = preprocess_dataset()
print(boxes)
display_boxes(labels, boxes, img_list)

#train_images = np.array(img_list)
#train_boxes = np.array(boxes)

classes = ['long sleeve dress', 'long sleeve outwear', 'long sleeve top', 
'short sleeve dress', 'short sleeve outwear', 'short sleeve top', 'short', 'skirt', 'sling', 'sling dress', 'trousers', 'vest', 'vest dress']

train_labels = []

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
  elif _:
      print("Non ha classe")
for label in labels:
    train_labels.append(change(label))
#train_labels = np.array(train_labels)


train_images, val_images, train_labels, \
val_labels, train_boxes, val_boxes = train_test_split( np.array(img_list), 
                np.array(train_labels), np.array(boxes), test_size = 0.2, 
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

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # CNNs for rgb images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=192, kernel_size=5)

        self.class_fc1 = nn.Linear(in_features=1728, out_features=240)
        self.class_fc2 = nn.Linear(in_features=240, out_features=120)
        self.class_out = nn.Linear(in_features=120, out_features=13)

        self.box_fc1 = nn.Linear(in_features=1728, out_features=240)
        self.box_fc2 = nn.Linear(in_features=240, out_features=120)
        self.box_out = nn.Linear(in_features=120, out_features=4)


    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv4(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv5(t)
        t = F.relu(t)
        t = F.avg_pool2d(t, kernel_size=4, stride=2)

        t = torch.flatten(t,start_dim=1)


        class_t = self.class_fc1(t)
        class_t = F.relu(class_t)

        class_t = self.class_fc2(class_t)
        class_t = F.relu(class_t)

        class_t = F.softmax(self.class_out(class_t),dim=1)

        box_t = self.box_fc1(t)
        box_t = F.relu(box_t)

        box_t = self.box_fc2(box_t)
        box_t = F.relu(box_t)

        box_t = self.box_out(box_t)
        box_t = F.torch.sigmoid(box_t)

        return [class_t,box_t]


print("Creata cnn")
model = Network()
model = model.to(device)
print("Passata al device --> cpu in questo caso")


def get_num_correct(preds, labels):
    return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=32, shuffle=True)

def train(model):
    # Defining the optimizer
    optimizer = optim.SGD(model.parameters(),lr = 0.1)
    num_of_epochs = 30
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
            x,y,z = x.to(device),y.to(device),z.to(device)
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            [y_pred,z_pred]= model(x)
            # Compute loss (here CrossEntropyLoss)
            class_loss = F.cross_entropy(y_pred, y)
            box_loss = F.mse_loss(z_pred, z)
            (box_loss + class_loss).backward()
            # class_loss.backward()
            optimizer.step()
            print("Train batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-train_start)/60, end='\r')
        model.eval()
        for batch, (x, y,z) in enumerate(valdataloader):
        	# Converting data from cpu to GPU if available to improve speed	
            x,y,z = x.to(device),y.to(device),z.to(device)
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            with torch.no_grad():
                [y_pred,z_pred]= model(x)
                
                # Compute loss (here CrossEntropyLoss)
                class_loss = F.cross_entropy(y_pred, y)
                box_loss = F.mse_loss(z_pred, z)
                # Compute loss (here CrossEntropyLoss)

            tot_loss += (class_loss.item() + box_loss.item())
            tot_correct += get_num_correct(y_pred, y)
            print("Test batch:", batch+1, " epoch: ", epoch, " ",
                  (time.time()-train_start)/60, end='\r')
        epochs.append(epoch)
        losses.append(tot_loss)
        print("Epoch", epoch, "Accuracy", (tot_correct)/2.4, "loss:",
              tot_loss, " time: ", (time.time()-train_start)/60, " mins")
        torch.save(model.state_dict(), "models/model_ep"+str(epoch+1)+".pth")

print("Creato train")    
#train(model)
print("Eseguito train")    



def preprocess(img, image_size = 256):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image



def postprocess(image, results):
    [class_probs, bounding_box] = results
    class_index = torch.argmax(class_probs)
    class_label = classes[class_index]
    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(256 * x1)
    x2 = int(256 * x2)
    y1 = int(256 * y1)
    y2 = int(256 * y2)
    print(x1, x2, y1, y2)
    # return the lable and coordinates
    return class_label, (x1,y1,x2,y2),torch.max(class_probs)*100



# We will use this function to make prediction on images.
def predict(image,  scale = 0.5):
    model = Network()
    model = model.to(device)
    model.load_state_dict(torch.load("models/model_ep29.pth"))
    model.eval()
    
    # Reading Image
    img  = cv2.imread(image)
    h = img.shape[0] / 256
    w = img.shape[1] / 256
    print(h, w)
    # # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(img)
    result = model(torch.permute(torch.from_numpy(processed_image).float(),(0,3,1,2)).to(device))

    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence = postprocess(image, result)
    print("Ciao")
    # Now annotate the image
    x1 = int(x1 * w)
    x2 = int(x2 * h)
    y1 = int(y1 * w)
    y2 = int(y2 * h)
    print(x1, x2, y1, y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 2)
    cv2.putText(
        img, 
        '{}, CONFIDENCE: {}'.format(label, confidence), 
        (30, int(35 * scale)), 
        cv2.FONT_HERSHEY_COMPLEX, scale,
        (200, 55, 100),
        2
        )

    # Show the Image with matplotlib
    #plt.figure(figsize=(10,10))
    plt.imshow(img[:,:,::-1])
    plt.show()


image = "../assignment_1/train/images/001820.jpg"
predict(image)