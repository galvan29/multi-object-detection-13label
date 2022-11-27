import os
import cv2
import json

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
    img_list = []

    h = 256
    w = 256
    image_dir = '/home/fdsa/Documents/assignment_1/train/'

    rows = os.listdir(image_dir + "images")
    cicle = 0
    for row in rows:
        stringa = str(row[0:6]) + ".json"
        with open(image_dir + "annotations/" + stringa, 'r') as f:
            data = json.loads(f.read())
            array = items(data)
            
            img_path = row
            img  = cv2.imread(os.path.join(image_dir + "images/",img_path))
            image = cv2.resize(img, (256, 256))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype("float") / 255.0
            img_list.append(image)
            for item in array:
                boxes.append(item['bounding_box'])
                labels.append(item['category_name'])

            cicle += 1

    return labels, boxes, img_list

preprocess_dataset()