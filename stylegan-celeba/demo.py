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
model=torch.load('stylegan-cele.pth')
model.eval()
root='../'
corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))

fw=open('result.txt','w')
f=open('list')
list=[]
for line in f:
 list.append(line.split(' ')[0])

model.eval()







print ('in dataset and cross-GAN')













corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))







cnt=0

gt=0
paths=glob.glob(root+'pngdata/data/style-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #if jpeg:
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     im=cv2.resize(im,(64,64))
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
     print (path, 'resize', pred)
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
paths=glob.glob(root+'pngdata/data/style-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #if jpeg:
     rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path, 'jpeg', pred)
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
paths=glob.glob(root+'pngdata/data/style-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #if jpeg:
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path, 'origin', pred)
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
paths=glob.glob(root+'pngdata/data/celeba-1024/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     #if jpeg:
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     im=cv2.resize(im,(64,64))
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
     print (path, 'resize', pred)
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
paths=glob.glob(root+'pngdata/data/celeba-1024/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     #if jpeg:
     rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path,  'jpeg', pred)
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
paths=glob.glob(root+'pngdata/data/celeba-1024/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #if jpeg:
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path, 'origin', pred)
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
paths=glob.glob(root+'pngdata/data/prog-gan-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     im=cv2.resize(im,(64,64))
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
     print (path, 'resize', pred)
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
paths=glob.glob(root+'pngdata/data/prog-gan-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path, 'jpeg', pred)
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
paths=glob.glob(root+'pngdata/data/prog-gan-cele/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)
     
     #im=cv2.resize(im,(64,64))
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
     print (path, 'origin', pred)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))




print ('Cross Dataset')





cnt=0
gt=0
paths=glob.glob(root+'pngdata/dcgan/*')

paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     h=im.shape[0]
     ims = np.zeros((1, 3, h, h))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)
     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1
 
fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()


corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))

cnt=0
gt=0
paths=glob.glob(root+'pngdata/dragan/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     h=im.shape[0]
     ims = np.zeros((1, 3, h, h))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)
     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1
 
fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()

cnt=0
gt=0
paths=glob.glob(root+'pngdata/stargan/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     h=im.shape[0]
     ims = np.zeros((1, 3, h, h))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)
     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1
 
fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()


corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))


gt=1
cnt=0
paths=glob.glob(root+'pngdata/celeba-lowres/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     h=im.shape[0]
     ims = np.zeros((1, 3, h, h))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)
     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()






corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))
cnt=0
gt=1
paths=glob.glob(root+'pngdata/data/ffhq/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1

     im=cv2.imread(path)
     #if jpeg:
     #rate=95
     #encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     #result,im=cv2.imencode('.jpg',im,encode_param)
     #im=cv2.imdecode(im,1)




     ims = np.zeros((1, 3, 1024, 1024))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)

     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()







cnt=0
corr=0.0
wrong=0.0

corrs = np.zeros((2,1))
wrongs = np.zeros((2,1))

gt=0
paths=glob.glob(root+'pngdata/data/style-ffhq/*')
paths.sort()
for path in paths:
     name=path.split('/')[-1].split('.')[0]

     cnt+=1
     im=cv2.imread(path)
     #kernel=3
     #noise=np.random.normal(0,3,(1024,1024,3)).astype('uint8')
     #im+=noise
     #im=cv2.GaussianBlur(im,(3,3),0)
     #if jpeg:
     '''rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     #im = cv2.resize(im, (512, 512))
     im=cv2.resize(im,(64,64))'''



     ims = np.zeros((1, 3, 1024, 1024))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]

     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()

     pred=np.argmax(output)
     print (path,  pred)
     if int(gt)==int(pred):
       corr+=1
       corrs[int(gt)] = corrs[int(gt)]+1
     else:
       wrong+=1
       wrongs[int(gt)] = wrongs[int(gt)] + 1

fw.write(str(corrs[0]/(corrs[0]+wrongs[0]+1)))
fw.write(str(corrs[1]/(corrs[1]+wrongs[1]+1)))
fw.flush()