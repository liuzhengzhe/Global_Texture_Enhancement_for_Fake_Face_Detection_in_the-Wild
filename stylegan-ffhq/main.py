
import numpy as np
import torch,os,random
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms#, models
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.utils.model_zoo as model_zoo
import resnet18_gram as resnet
import os,glob
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.autograd import Variable
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    #transforms.Scale(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

root='./'

def default_loader(path):

    size = random.randint(64, 256)

    im=cv2.imread(path)
    im=cv2.resize(im,(size,size))
    im=cv2.resize(im,(512,512))
    ims=np.zeros((3,512,512))
    ims[0,:,:]=im[:,:,0]
    ims[1,:,:]=im[:,:,1]
    ims[2,:,:]=im[:,:,2]
    img_tensor=torch.tensor(ims.astype('float32'))
    
    return img_tensor

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(' ')[0]) for line in lines]
            self.img_label = [int(line.strip().split(' ')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
 
image_datasets = customData(img_path='',txt_path=('list')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

dataloders =  torch.utils.data.DataLoader(image_datasets,
                                                 batch_size=14,
                                                 shuffle=True) 



 
test_datasets = customData(img_path='',txt_path=('list')) #,
                                    #data_transforms=data_transforms,
                                    #dataset=x) for x in ['train', 'val']}

testloader =  torch.utils.data.DataLoader(test_datasets,
                                                 batch_size=1,
                                                 shuffle=False) 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet.resnet18(pretrained=True) #resnet18_gram.resnet18() #pretrained=True)
resnetinit=torch.load('stylegan-ffhq.pth')
model.load_state_dict(resnetinit.state_dict(),strict=False)



criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
#model=torch.load('aerialmodel.pth')
model.to(device)


def test0(model):
  model.eval()
  gt=0
  corr=0
  for i in range(0,100):
     im=cv2.imread(root+'pngdata/stargan/'+str(i).zfill(6)+'.png')
     ims = np.zeros((1, 3, 128, 128))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==0:
        corr+=1
  return corr/100.0

def test1(model):
  model.eval()

  gt=0
  corr=0
  for i in range(10000,10100):
     im=cv2.imread(root+'pngdata/data/prog-gan-cele/'+str(i).zfill(6)+'.png')
     '''rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     if random.randint(0,1)==0:
       im=cv2.resize(im,(64,64))
     im=cv2.resize(im,(512,512))'''
     ims = np.zeros((1, 3, 1024, 1024))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==0:
        corr+=1
  return corr/50.0

def test2(model):
  model.eval()
  gt=0
  corr=0
  for i in range(0,100):
     im=cv2.imread(root+'pngdata/dcgan/'+str(i).zfill(6)+'.png')
     ims = np.zeros((1, 3, 64, 64))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==0:
        corr+=1
  return corr/100.0


def test3(model):
  model.eval()
  gt=0
  corr=0
  paths=glob.glob(root+'pngdata/celeba-low-res/20????.png')
  for path in paths:
     im=cv2.imread(path)
     ims = np.zeros((1, 3, 178, 178))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('1',pred)
     if pred==1:
        corr+=1
  return corr/131.0

def test4(model):
  model.eval()
  gt=0
  corr=0
  for i in range(100,150):
     im=cv2.imread(root+'pngdata/data/ffhq/'+str(i).zfill(6)+'.png')
     rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     if random.randint(0,1)==0:
       im=cv2.resize(im,(64,64))
     im=cv2.resize(im,(512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==1:
        corr+=1
  return corr/50.0

def test5(model):
  model.eval()
  gt=0
  corr=0
  for i in range(100,150):
     im=cv2.imread(root+'pngdata/data/style-ffhq/'+str(i).zfill(6)+'.png')
     rate=95
     encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),rate]
     result,im=cv2.imencode('.jpg',im,encode_param)
     im=cv2.imdecode(im,1)
     if random.randint(0,1)==0:
       im=cv2.resize(im,(64,64))
     im=cv2.resize(im,(512,512))
     ims = np.zeros((1, 3, 512,512))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==0:
        corr+=1
  return corr/50.0


def test6(model):
  model.eval()

  gt=0
  corr=0
  for i in range(10000,10050):
     im=cv2.imread(root+'pngdata/data/style-cele/'+str(i).zfill(6)+'.png')
     ims = np.zeros((1, 3, 1024, 1024))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==0:
        corr+=1
  return corr/50.0



def test7(model):
  model.eval()

  gt=0
  corr=0
  for i in range(10000,10040):
     im=cv2.imread(root+'pngdata/data/celeba-1024/'+str(i)+'.jpg')
     ims = np.zeros((1, 3, 1024, 1024))
     ims[0, 0, :, :] = im[:, :, 0]
     ims[0, 1, :, :] = im[:, :, 1]
     ims[0, 2, :, :] = im[:, :, 2]
     image_tensor =torch.tensor(ims).float()
     inputs = Variable(image_tensor).float().cuda()
     output = model(inputs)
     output=output.detach().cpu().numpy()
     pred=np.argmax(output)
     #print ('0',pred)
     if pred==1:
        corr+=1
  return corr/40.0

lr=0.00001
for param_group in optimizer.param_groups:
  param_group['lr']=lr


def get_lr(optimizer):
  lr=[]
  for param_group in optimizer.param_groups:
    lr+=[param_group['lr']]
  return lr
epochs = 5
steps = 0
running_loss = 0
print_every = 1000
train_losses, test_losses = [], []


maxi=0
for epoch in range(epochs):

    for inputs,labels in dataloders:
        model.train()
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        print (loss,'loss',epoch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        r0=test0(model)
        r1=test1(model)
        r2=test2(model)
        r3=test3(model)
        r4=test4(model)
        r5=test5(model)
        r6=test6(model)
        r7=test7(model)
        #r8=test8(model)
        score=r0+r1+r2+r3*2+r4+r5+r6+r7
        if score>maxi:
           maxi=score
   
           torch.save(model, 'aerialmodel'+str(len(glob.glob('*.pth')))+'.pth')

        #if steps%100==0:
        #  torch.save(model, 'aerialmodel'+str(len(glob.glob('*.pth')))+'.pth')




