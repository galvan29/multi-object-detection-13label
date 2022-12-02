from network import Network
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torch.optim as optim
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img, image_size):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0 
    image = np.expand_dims(image, axis=0) 
    
    return image

def postprocess(image, results, classes, scale):
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

def predict(image, number_model, scale, showfinalimage, saveimage, classes, savedir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Network()
    model = model.to(device)
    model.load_state_dict(torch.load(savedir+"/model_ep"+str(number_model)+".pth"))
    model.eval()
    
    # Reading Image
    img  = cv2.imread(image)
    h = img.shape[0] / scale
    w = img.shape[1] / scale

    # Before we can make a prediction we need to preprocess the image.
    processed_image = preprocess(img, scale)
    result = model(torch.permute(torch.from_numpy(processed_image).float(),(0,3,1,2)).to(device))

    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence, label1, (x3, y3, x4, y4), confidence1 = postprocess(image, result, classes, scale)

    img = cv2.resize(img, (scale, scale))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 100), 1)
    cv2.rectangle(img, (x3, y3), (x4, y4), (255, 0, 100), 1)

    print("Informazioni")
    print(label, confidence.item())
    print(label1, confidence1.item())

    # Showing and saving predicted
    plt.imshow(img[:,:,::-1])
    if saveimage:
        plt.savefig("./" + image[-10:])
    if showfinalimage:
        plt.show()