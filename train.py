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
    
    parser.add_argument('data_directory', default = 'ImageClassifier/flowers/',help ='data directory')
    parser.add_argument('--save_dir',help = 'directory used to save model')
    parser.add_argument('--arch',help = 'choose between different archs')
    parser.add_argument('--learning_rate',help = 'how fast the network learns')
    parser.add_argument('--hidden_units',help = 'number of hidden units')
    parser.add_argument('--epochs',help = 'number of cycles (epochs)')
    parser.add_argument('--gpu',action = 'store_true', help = 'gpu mode')
    
    return parser.parse_args()


def process(data_direc):
    train_dir, valid_dir, test_dir = data_direc
    
    trans = transforms.Compose([transforms.Resize(225),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    image_datasets = datasets.ImageFolder(train_dir ,transform = trans)
    valid_datasets = datasets.ImageFolder(valid_dir ,transform = trans)
    test_datasets = datasets.ImageFolder(test_dir ,transform = trans)
    
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle= True)
    valloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle = True)
    
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    ldrs = {'train':dataloaders,'valid':valloaders,'test':testloaders,'labels':cat_to_name}
    return ldrs
    

    

def find_data():
    train_dir = args.data_directory + '/train'
    valid_dir = args.data_directory + '/valid'
    test_dir = args.data_directory + '/test'
    
    data_direc = [train_dir, valid_dir,test_dir]
    return process(data_direc)


def build(data):
    
    if(args.arch is None):
        arch = 'vgg'
    else:
        arch = args.arch

    
    if (arch == 'vgg'):
        model = models.vgg19(pretrained = True)
        input_n = 25088
    elif(arch == 'desnsenet'):
        model = models.densenet121(pretrained = True)
        input_n = 1024
    
    if (args.hidden_units is None):
        hidden_units = 4096
    else:
        hidden_units = args.hidden_units

    
    hidden_units = int(hidden_units)
    
    for p in model.parameters():
        p.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('dropout',nn.Dropout(0.5)),
                                            ('fc1',nn.Linear(input_n,hidden_units)),
                                            ('relu1',nn.ReLU()),
                                            ('fc2',nn.Linear(hidden_units,102)),
                                            ('output',nn.LogSoftmax(dim = 1))]))
    
    model.classifier = classifier
    
    return model
        
def valid(model,validloaders,criterion):
    tst_loss=0
    acc = 0

    for images,labels in validloaders:

        images,labels = images.to(device),labels.to(device)
        output = model.forward(images)
        tst_loss +=criterion(output,labels).item()

        prob = torch.exp(output)
        eq = (labels.data == prob.max(dim=1)[1])
        acc += eq.type(torch.FloatTensor).mean()
    return tst_loss,acc
    
def train(model,data):
    
    dataloaders = data['train']
    valloaders = data['valid']
    testloaders = data['test']
    
    if(args.learning_rate is None):
        learnrate = 0.001
    else:
        learnrate = args.learning_rate
        
    if(args.epochs is None):
        epochs = 3
    else:
        epochs = args.epochs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = int(epochs)

    learnrate = float(learnrate)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    print_every = 5
    steps = 0
    model.to(device)

    model.train()
    for e in range(epochs):
        run_loss = 0 
        for images, labels in dataloaders:

            optimizer.zero_grad()
            images, labels = images.to(device),labels.to(device)
            steps+=1
            out = model.forward(images)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()

            run_loss+=loss.item()

            if steps%print_every == 0:

                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = valid(model,valloaders,criterion)

                print("Epoch: {}/{}".format(e+1,epochs),"Train Loss:                                                                                                   {:.4f}".format(run_loss/print_every),"Validation Loss:                                                                                           {:.4f}".format(test_loss/len(valloaders)),"Validation Accuracy {:.4f}".format(accuracy/len(valloaders)))

                run_loss = 0
                model.train()
 

    return model

        
def save_m(model):
    save_dir = args.save_dir
    
    checkpoint = {'model':model.cpu(),
                  'classifier':model.classifier,
                  #'features':model.features,
                  'state_dict':model.state_dict()}
    
    torch.save(checkpoint,save_dir)
    
    
def create_m():
    data = find_data()
    model = build(data)
    model = train(model,data)
    save_m(model)
    return None



def main():
    print("Start Model")
    global args
    args = get_args()
    create_m()
    print("Done")
    
    return None


main()