import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision

import logging
from torch.utils.tensorboard import SummaryWriter

#Takes a list of image tensors and outputs them as a normalized torch tensor
def encode(ims):
    arr = []
    for im in ims:
        im = im.clone().detach()
        im = im / 255    #outputs into [0,1]
        im = im*2 - 1  #outputs into [-1,1]
        arr.append(im)
    return torch.stack(arr) 
    

#Decode inpute torch tensor into an image
def decode(img):
    img = (img.clamp(-1,1) + 1)/2  #outputs into [0,1]
    img = torch.round(img * 255) #outputs into [0,255]
    img = img.type(torch.uint8)
    #img = img.view(3,64,64)
    return img


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



def save_images(images, path):
    for i in range(len(images)):
        impath = path + str(i) + ".jpg"
        torchvision.io.write_jpeg(images[i], impath)


import os
def setup_logging(run_name):
    os.makedirs("models_weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models_weights", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


