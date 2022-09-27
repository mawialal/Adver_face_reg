from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from PIL import Image
import cv2
from matplotlib.pyplot import imshow
import torch.optim as optim
from torch import nn

import torchvision.transforms as transforms
from PIL import Image
import cv2
from matplotlib.pyplot import imshow
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def org_pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        y = torch.LongTensor([y]).to('cuda')
        y_targ = torch.LongTensor([y_targ]).to('cuda')
        yp = model(X + delta)
        loss = (yp[:,y_targ] - yp.gather(1,y[:,None])[:,0]).sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def get_avd_class(model, avd_file_path):
    ## Get class of avd image
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )
    
    
    avd_img = Image.open(avd_file_path)
    x_aligned, prob = mtcnn(avd_img, return_prob=True)
    aligned = torch.stack([x_aligned]).to(device)
    prob = model(aligned)
    avd_class = torch.argmax(prob).item()
    
    return avd_class

def create_targeted_fake(model, file_path, avd_file_path):
    # Function for generating adveserial examples
    
    ## Get class of avd image
    avd_class = get_avd_class(model,avd_file_path)
    
    ## Crop 
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )

    img = Image.open(file_path)
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    left = int(boxes[0][0])
    top = int(boxes[0][1])
    right = int(boxes[0][2])
    bottom = int(boxes[0][3])
    #img=cv2.imread(file_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
    width=right-left
    height=bottom-top
    img_crop=img[top:top+height,left:left+width]
    
    
    crop_height = img_crop.shape[0]
    crop_width = img_crop.shape[1]
    
    ## Resize to input size
    img_crop = cv2.resize(img_crop, (224,224), interpolation = cv2.INTER_AREA)
  
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)    
    img_pil = Image.fromarray(img_crop)
    
    
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]


    im = transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                         std = std)])(img_pil)

    
    aligned = torch.stack([im]).to('cuda')
    prob = model(aligned)
    y = torch.argmax(prob).item()
    
    delta = org_pgd_linf_targ(model, aligned, y, epsilon=0.2, alpha=1e-2, num_iter= 8, y_targ=avd_class)
    
    print(delta)
    
    # Add adveserail noise to image
    pref_image = aligned + delta
    
    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv*torch.ones(A.shape))
        A = torch.min(A, maxv*torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
    transforms.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
    transforms.Lambda(clip),
    transforms.ToPILImage(),
    transforms.CenterCrop(224)
    ])
    
    image = np.asarray(tf(pref_image.cpu()[0]))
    
    image = cv2.resize(image, (crop_width,crop_height), interpolation = cv2.INTER_AREA)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[top:top+crop_height,left:left+crop_width] = image
              
   
    #plt.figure()
    #plt.imshow(img)
    #title = "Orignal : "+ str(y) + " : "  + "  Adversarial example : "+  str(avd_class)
    #plt.title(title)
    #plt.show()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite(file_path.split('.')[0]+"avd.jpeg",img)
    return img