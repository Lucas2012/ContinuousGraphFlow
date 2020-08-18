import os,sys
import numpy as np
import PIL
import argparse

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * 255 + noise
    x = x / 256
    return x

class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)



class CelebaHQDataset(Dataset):
  def __init__(self,image_dir,split_dir,split,image_size=512,normalize=False,noise=True):
    self.image_dir = image_dir
    self.split_dir = split_dir
    self.split = split
    self.image_size = image_size

    self.normalize = normalize
    self.noise = noise
    self.image_paths = [] 

    self.split_file = os.path.join(split_dir,split+'.txt')
    with open(self.split_file) as f:
      content = f.readlines()
    self.image_paths = [x.strip() for x in content]
    
    transform = [Resize(image_size), T.ToTensor()]
    if self.normalize:
      transform.append(imagenet_preprocess())
    if self.noise:
      transform.append(add_noise)
    self.transform = T.Compose(transform)

    self.transform = T.Compose(transform)

  def __len__(self):
    with open(self.split_file) as f:
      content = f.readlines()

    content = [x.strip() for x in content]

    return len(content)

  def __getitem__(self,index):
    path = self.image_paths[index]

    with open(os.path.join(self.image_dir,path),'rb') as f:
      with PIL.Image.open(f) as img:
        img_W,img_H  = img.size
        img = img.convert('RGB') 
        img = self.transform(img)


    return img,path


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_dir',type=str,default='data/celeba_hq/imgs')
  parser.add_argument('--split_dir',type=str,default='data/celeba_hq/splits')
  parser.add_argument('--split',type=str,default='train')
  parser.add_argument('--image_size',type=int,default=128)
  parser.add_argument('--batch_size',type=int,default=32)
  parser.add_argument('--workers',type=int,default=8)
  
  args = parser.parse_args()
  train_dataset = CelebaHQDataset(args.image_dir,args.split_dir,'train',args.image_size)
  train_dataloader= DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)


  test_dataset = CelebaHQDataset(args.image_dir,args.split_dir,'test',args.image_size)
  test_dataloader= DataLoader(test_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)


  for i,data in enumerate(train_dataloader):
    print(i,data.size())
 


