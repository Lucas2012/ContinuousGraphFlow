'''
Dataloader for Visual Genome dataset
Each data return the following information:
Class label, position boxes, relationships [relationship_subjects, relationship_predicates, relationship_objects]
'''
import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

import numpy as np
import h5py
import json
import PIL
import argparse
from lib.utils import int_tuple, bool_flag


torch.backends.cudnn.benchmark = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

def imagenet_preprocess():
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


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


class VgSceneGraphDataset(Dataset):
  def __init__(self, vocab, h5_path, image_dir, image_size=(256, 256),
               normalize_images=True, max_objects=10, max_samples=None,
               include_relationships=True, use_orphaned_objects=True):
    super(VgSceneGraphDataset, self).__init__()

    self.image_dir = image_dir
    self.image_size = image_size
    self.vocab = vocab
    self.num_objects = len(vocab['object_idx_to_name'])
    self.use_orphaned_objects = use_orphaned_objects
    self.max_objects = max_objects
    self.max_samples = max_samples
    self.include_relationships = include_relationships

    transform = [Resize(image_size), T.ToTensor()]
    if normalize_images:
      transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)

    self.data = {}
    with h5py.File(h5_path, 'r') as f:
      for k, v in f.items():
        if k == 'image_paths':
          self.image_paths = list(v)
        else:
          self.data[k] = torch.IntTensor(np.asarray(v))

  def __len__(self):
    num = self.data['object_names'].size(0)
    if self.max_samples is not None:
      return min(self.max_samples, num)
    return num

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - objs: LongTensor of shape (O,)
    - position boxes: FloatTensor of shape (O, 4) giving boxes for objects in
      (x0, y0, x1, y1) format, in a [0, 1] coordinate system.
    - relationship triples [relationship_subjects, relationship_predicates, relationship_objects]: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
      means that (objs[i], p, objs[j]) is a triple.
    """
    try:
      img_path = os.path.join(self.image_dir, self.image_paths[index].decode("utf-8"))
    except:
      img_path = os.path.join(self.image_dir, self.image_paths[index])

    with open(img_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    H, W = self.image_size

    # Figure out which objects appear in relationships and which don't
    obj_idxs_with_rels = set()
    obj_idxs_without_rels = set(range(self.data['objects_per_image'][index].item()))
    for r_idx in range(self.data['relationships_per_image'][index]):
      s = self.data['relationship_subjects'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      obj_idxs_with_rels.add(s)
      obj_idxs_with_rels.add(o)
      obj_idxs_without_rels.discard(s)
      obj_idxs_without_rels.discard(o)

    obj_idxs = list(obj_idxs_with_rels)
    obj_idxs_without_rels = list(obj_idxs_without_rels)
    if len(obj_idxs) > self.max_objects - 1:
      obj_idxs = random.sample(obj_idxs, self.max_objects)
    if len(obj_idxs) < self.max_objects - 1 and self.use_orphaned_objects:
      num_to_add = self.max_objects - 1 - len(obj_idxs)
      num_to_add = min(num_to_add, len(obj_idxs_without_rels))
      obj_idxs += random.sample(obj_idxs_without_rels, num_to_add)
    O = len(obj_idxs) + 1

    objs = torch.LongTensor(O).fill_(-1)

    boxes = torch.FloatTensor([[0, 0, 1, 1]]).repeat(O, 1)
    obj_idx_mapping = {}
    for i, obj_idx in enumerate(obj_idxs):
      objs[i] = self.data['object_names'][index, obj_idx].item()
      x, y, w, h = self.data['object_boxes'][index, obj_idx].tolist()
      x0 = float(x) / WW
      y0 = float(y) / HH
      x1 = float(x + w) / WW
      y1 = float(y + h) / HH
      boxes[i] = torch.FloatTensor([x0, y0, x1, y1])
      obj_idx_mapping[obj_idx] = i

    # The last object will be the special __image__ object
    objs[O - 1] = self.vocab['object_name_to_idx']['__image__']

    triples = []
    for r_idx in range(self.data['relationships_per_image'][index].item()):
      if not self.include_relationships:
        break
      s = self.data['relationship_subjects'][index, r_idx].item()
      p = self.data['relationship_predicates'][index, r_idx].item()
      o = self.data['relationship_objects'][index, r_idx].item()
      s = obj_idx_mapping.get(s, None)
      o = obj_idx_mapping.get(o, None)
      if s is not None and o is not None:
        triples.append([s, p, o])

    # Add dummy __in_image__ relationships for all objects
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])

    triples = torch.LongTensor(triples)
    return  objs, boxes, triples


def vg_collate_fn(batch):
  """
  Collate function to be used when wrapping a VgSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:
  - imgs: FloatTensor of shape (N, C, H, W)
  - objs: LongTensor of shape (O,) giving categories for all objects
  - boxes: FloatTensor of shape (O, 4) giving boxes for all objects
  - triples: FloatTensor of shape (T, 3) giving all triples, where
    triples[t] = [i, p, j] means that [objs[i], p, objs[j]] is a triple
  - obj_to_img: LongTensor of shape (O,) mapping objects to images;
    obj_to_img[i] = n means that objs[i] belongs to imgs[n]
  - triple_to_img: LongTensor of shape (T,) mapping triples to images;
    triple_to_img[t] = n means that triples[t] belongs to imgs[n].
  """
  # batch is a list, and each element is (image, objs, boxes, triples) 
  all_imgs, all_objs, all_boxes, all_triples = [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  max_graph_size = 0
  for i, ( objs, boxes, triples) in enumerate(batch):
    #all_imgs.append(img[None])
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    triples = triples.clone()
    #triples[:, 0] += obj_offset
    #triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O
    if max_graph_size < O:
      max_graph_size = O

  #all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_objs, all_boxes, all_triples,
         all_obj_to_img, all_triple_to_img,
         max_graph_size, len(batch))
  out = build_graph(*out)
  return out


def build_graph(all_objs, all_boxes, all_triples, all_obj_to_img, all_triple_to_img, max_graph_size, batch_size):
  graphs       = -1 * torch.ones(batch_size, max_graph_size)
  graph_matrix = -1 * torch.ones(batch_size, max_graph_size, max_graph_size)
  graph_boxes  = -1 * torch.ones(batch_size, max_graph_size, 4)
  graph_masks  = torch.zeros(batch_size, max_graph_size)
  for batch_idx in all_obj_to_img.unique():
    indices = (all_obj_to_img==int(batch_idx.item()))
    objs    = all_objs[indices]
    boxes   = all_boxes[indices]
    triples = all_triples[(all_triple_to_img==int(batch_idx.item())).unsqueeze(1).expand(-1, 3)].view(-1, 3)
    graphs[batch_idx, :len(objs)]          = objs
    graph_boxes[batch_idx, :len(boxes), :] = boxes.view(-1, 4)
    graph_masks[batch_idx, :len(objs)] = 1
    for p_i in range(0, len(triples)):
      graph_matrix[batch_idx][triples[p_i][0], triples[p_i][2]] = triples[p_i][1]

  return graphs.long(), graph_matrix, graph_boxes, graph_masks, all_objs, all_triples, all_obj_to_img, all_triple_to_img


def build_vg_dsets(args):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.train_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.imagesize,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
  }
  train_dset = VgSceneGraphDataset(**dset_kwargs)
  print("train_dset[0] ", train_dset[0])
  iter_per_epoch = len(train_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  dset_kwargs['h5_path'] = args.val_h5
  del dset_kwargs['max_samples']
  val_dset = VgSceneGraphDataset(**dset_kwargs)
  
  return vocab, train_dset, val_dset


def build_loaders(dataset, batch_size, num_workers, shuffle, pin_memory, drop_last, collate_fn):
  
  loader_kwargs = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'shuffle': shuffle,
    'pin_memory': pin_memory,
    'collate_fn': collate_fn,
    'drop_last': drop_last
  }
  loader = DataLoader(dataset, **loader_kwargs)
  
  return loader
    

if __name__ == '__main__':
    VG_DIR = os.path.expanduser('data/vg')
    COCO_DIR = os.path.expanduser('data/coco')
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])
   
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_iterations', default=1000000, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
   
    # Switch the generator to eval mode after this many iterations
    parser.add_argument('--eval_mode_after', default=100000, type=int)
   
    # Dataset options common to both VG and COCO
    parser.add_argument('--image_size', default='64,64', type=int_tuple)
    parser.add_argument("--imagesize", default='64,64', type=int_tuple)
    parser.add_argument('--num_train_samples', default=None, type=int)
    parser.add_argument('--num_val_samples', default=1024, type=int)
    parser.add_argument('--shuffle_val', default=True, type=bool_flag)
    parser.add_argument('--loader_num_workers', default=4, type=int)
    parser.add_argument('--include_relationships', default=True, type=bool_flag)
   
    # VG-specific options
    parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
    parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
    parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
    parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
    parser.add_argument('--max_objects_per_image', default=10, type=int)
    parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)
 
    args = parser.parse_args()
    vocab, train_dset, test_dset = build_vg_dsets(args)

    train_loader = build_loaders(train_dset, args.batch_size, 4, True, True, True)
    test_loader  = build_loaders(test_dset, args.batch_size, 4, False, True, True)

    train_iter  = iter(train_loader)
    train_batch = next(train_iter)

    graphs, graph_matrix, graph_boxes, graph_masks = train_batch

    print(graphs, graph_matrix, graph_boxes, graph_masks)
