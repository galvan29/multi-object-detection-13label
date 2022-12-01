import os
import dataloader 
from dataloader import Dataset
from dataloader import ValDataset
from dataloader import process_dataset as dataset
from network import *
import sys


# Settable parameters
scale           = 160 
samples         = 100 
active_train    = True
num_of_epochs   = 5
image_dir       = './assignment_1/train/'
load_model      = False
number_model    = 141
lr              = 0.0001

# specigic
changes = {
	"long sleeve dress" : 0,
    "long sleeve outwear" : 1,
    "long sleeve top" : 2,
    "short sleeve dress" : 3,
    "short sleeve outwear" : 4,
    "short sleeve top" : 5,
    "shorts" : 6,
    "skirt" : 7,
    "sling" : 8,
    "sling dress" : 9,
    "trousers" : 10,
    "vest" : 11,
    "vest dress" : 12
}

if __name__ == "__main__":
    print("usage: main.py scale samples active_train num_of_epochs load_model number_model lr")
    print(f"args: {sys.argv}")
    """
    scale           = int(sys.argv[0])
    samples         = int(sys.argv[1])
    active_train    = int(sys.argv[2])
    num_of_epochs   = int(sys.argv[3])
    load_model      = int(sys.argv[4])
    number_model    = int(sys.argv[5])
    lr              = int(sys.argv[6])
    """

    # Loading datase 
    labels, boxes, images, train_images, val_images, train_labels, val_labels, train_boxes, val_boxes = dataset(
        image_dir=image_dir,
        samples=samples,
        scale=scale,
        active_train=active_train,
        changes=changes
    ) 

    datas = Dataset(train_images, train_labels, train_boxes)
    valdataset = ValDataset(val_images, val_labels, val_boxes)

    # Train model
    if active_train:
        train(
            num_of_epochs=num_of_epochs,
            lr=lr,
            dataset=datas,
            valdataset=valdataset,
            samples=samples
        )

    print(datas)
    