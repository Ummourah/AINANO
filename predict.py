import os
import argparse 
import torch 
import json
from PIL import Image
from torchvision import datasets,transforms,models
from collections import OrderedDict
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('input',help = 'image operated on')
    parser.add_argument('m_checkpoint',help = 'model used')
    parser.add_argument('--top_k',help = 'number of predictions to show')
    parser.add_argument('--category_names',help = 'file contains names')
    parser.add_argument('--gpu',action = 'store_true', help = 'gpu mode')
    
    return parser.parse_args()



def load_m():
    info = torch.load(args.m_checkpoint)
    model = info['model']
    model.classifier = info['classifier']
    model.load_state_dict(info['state_dict'])
    
    return model


def process_image(image):
    img = Image.open(image)
    trans = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image = trans(img)
    #np_image = np.array(image)


    return image

def classify(image_path, topk=5):
    top_k = int(top_k)
    
    model.to('cuda')

    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    model = load_m()
    with torch.no_grad():
        output = model.forward(image.cuda())
        
    ps = torch.exp(output)
    probs,classes = ps.topk(topk)
    probs = probs.cpu().numpy()[0]
    classes = classes.cpu().numpy()[0]
    
    probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
    
    results = zip(probs,classes)
    
    return results


def read_cat():
    cat_file = args.category_names
    
    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)
        
return cat_to_name



def res_display(results):
    
    cat_file = read_cat()
    i = 0 
    for p,c in results:
        i++
        ctgry = cat_file.get(str(c),'None')
        print("class label: {} , class category: {} ".format(c,ctgry))
        
    return None
    
    
def main():
    global args
    args = get_args()
    
    if (args.top_k is None):
        top_k = 5
        
    else:
        top_k = args.top_k
        
    top_k = args.top_k
    image_path = args.input
    
    res = classify(image_path, top_k)
    res_display(res)
    
    return None


main()
    
