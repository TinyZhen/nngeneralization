# https://github.com/weiaicunzai/pytorch-cifar100/blob/2149cb57f517c6e5fa7262f958652227225d125b
# /models/vgg.py

import torch.nn as nn
from torchvision import models


cfg = {
  'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512,
        512, 'M']
}


class VGG(nn.Module):
  def __init__(self, features, num_class=10):
    super().__init__()
    self.features = features

    self.classifier = nn.Sequential(
      nn.Linear(25088, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      # nn.Linear(4096, 4096),
      # nn.BatchNorm1d(4096),
      # nn.ReLU(inplace=True),
      # nn.Dropout(),
      # nn.Linear(4096, 4096),
      # nn.BatchNorm1d(4096),
      # nn.ReLU(inplace=True),
      # nn.Dropout(),
      nn.Linear(4096, num_class)
      
    )

  def forward(self, x):
      # print(f'Input shape: {x.shape}')  # Print input shape
      x = self.features(x)
      # print(f'After features shape: {x.shape}')  # Print after features
      
      x = x.view(x.size(0), -1)  # Flatten for the classifier
      # print(f'After flatten shape: {x.shape}')  # Print after flatten
      x = self.classifier(x)
      # print(f'After classifier shape: {x.shape}')  # Print after classifier
      
      return x


def make_layers(cfg, batch_norm=False):
  layers = []

  input_channel = 3
  for l in cfg:
    if l == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      continue

    layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

    if batch_norm:
      layers += [nn.BatchNorm2d(l)]

    layers += [nn.ReLU(inplace=True)]
    input_channel = l

  return nn.Sequential(*layers)


def vgg11_bn(num_class):
  return VGG(make_layers(cfg['A'], batch_norm=True), num_class)


def vgg13_bn(num_class):
  return VGG(make_layers(cfg['B'], batch_norm=True), num_class)


def vgg16_bn(num_class):
  return VGG(make_layers(cfg['D'], batch_norm=True), num_class)


def vgg19_bn(num_class):
  return VGG(make_layers(cfg['E'], batch_norm=True), num_class)

def freeze_vgg_features(model):
    for param in model.features.parameters():
        param.requires_grad = False

def load_pretrained_vgg16(model,num_classes):
    

  # Modify the classifier by extending it with two extra layers
  new_classifier = nn.Sequential(
      nn.Linear(25088, 4096),  # Original layer (VGG16 has this as its final FC layer)
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.BatchNorm1d(4096),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      # nn.Linear(2048, 2048),
      # nn.BatchNorm1d(2048),
      # nn.ReLU(inplace=True),
      # nn.Dropout(),
      # nn.Linear(2048, 2048),  
      # nn.BatchNorm1d(2048),
      # nn.ReLU(inplace=True),
      # nn.Dropout(),
      nn.Linear(4096, num_classes),  # Add the final layer for 100 classes (CIFAR-100)
  )

# Assign the new classifier to the model
  model.classifier = new_classifier
  return model
