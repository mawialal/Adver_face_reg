import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
#from torch.autograd.gradcheck import zero_gradients

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
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
from vgg import VGG_16
#import dlib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
labels = open(os.path.join('names.txt'), 'r').read().split('\n')

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grand.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image
    
def create_fake(model, file_path):
    # Function for generating adveserial examples
    # Download Dlib CNN face detector
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
    
    
    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, model)
    
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
    
    image = np.asarray(tf(pert_image.cpu()[0]))
    
    image = cv2.resize(image, (crop_width,crop_height), interpolation = cv2.INTER_AREA)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[top:top+crop_height,left:left+crop_width] = image
    original_name = str(labels[label_orig])
    perf_name = str(labels[label_pert])
   
    return img,original_name,perf_name
