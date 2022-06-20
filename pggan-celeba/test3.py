import sys
import numpy as np
import torch,os,random,glob
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms#, models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.utils.model_zoo as model_zoo
import resnet18_gram as resnet
import os
#import scipy.io
from torchvision.transforms import transforms

gt=0
model=torch.load('pggan-celeba.pth')
model.eval()

corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))

fw=open('result3.txt','w')
f=open('list')
list=[]
for line in f:
 list.append(line.split(' ')[0])

model.eval()
corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0

gt=0
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/style-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im = cv2.GaussianBlur(im,(25,25),0)
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))



corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0

gt=0
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/style-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im=im.astype('float')
     noise = np.random.normal(0, 5, im.shape)
     im+=noise
     im[np.where(im<0)]=0
     im[np.where(im>255)]=255
     im=im.astype('uint8')
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))




corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0

gt=1
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/celeba-1024/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im = cv2.GaussianBlur(im,(25,25),0)
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))


corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))

cnt=0
gt=1
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/celeba-1024/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im=im.astype('float')
     noise = np.random.normal(0, 5, im.shape)
     im+=noise
     im[np.where(im<0)]=0
     im[np.where(im>255)]=255
     im=im.astype('uint8')
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))





corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0
f=open('list')
list=[]
for line in f:
 list.append(line.split(' ')[0])
gt=0
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/prog-gan-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im = cv2.GaussianBlur(im,(25,25),0)
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))



corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0
f=open('list')
list=[]
for line in f:
 list.append(line.split(' ')[0])
gt=0
paths=glob.glob('/media/lzz/LENOVO_USB_HDD/pngdata/data/prog-gan-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]
     if path in list:
      continue
     if int(name)>10000 and int(name)<10100:
      continue
     if cnt>10000:
      break
     cnt+=1

     im=cv2.imread(path)
     im=im.astype('float')
     noise = np.random.normal(0, 5, im.shape)
     im+=noise
     im[np.where(im<0)]=0
     im[np.where(im>255)]=255
     im=im.astype('uint8')
     im = cv2.resize(im, (512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))




