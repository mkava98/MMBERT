from PIL import Image
import torch
import torch.nn as nn
import timm
import requests
from torchvision import transforms, models

import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# print(torch.__version__)
# should be 1.8.0


model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
model2=models.resnet152(pretrained=True)
# model.eval()


relu = nn.ReLU()
# conv2 = nn.Conv2d(2048, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
# gap2 = nn.AdaptiveAvgPool2d((1,1))
# conv3 = nn.Conv2d(1024, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
# gap3 = nn.AdaptiveAvgPool2d((1,1))
# conv4 = nn.Conv2d(512, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
# gap4 = nn.AdaptiveAvgPool2d((1,1))
# conv5 = nn.Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
# gap5 = nn.AdaptiveAvgPool2d((1,1))
# conv7 = nn.Conv2d(64, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
# gap7 = nn.AdaptiveAvgPool2d((1,1))


conv2 = nn.Conv2d(196, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
gap2 = nn.AdaptiveAvgPool2d((1,1))
conv3 = nn.Conv2d(1024, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
gap3 = nn.AdaptiveAvgPool2d((1,1))
conv4 = nn.Conv2d(512, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
gap4 = nn.AdaptiveAvgPool2d((1,1))
conv5 = nn.Conv2d(256, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
gap5 = nn.AdaptiveAvgPool2d((1,1))
conv7 = nn.Conv2d(64, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
gap7 = nn.AdaptiveAvgPool2d((1,1))

transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
img = transform(img)[None,]
print("image size:", img.size())
# out = model(img)



modules2 = list(model.children())[:-4]
# print("children -2222",list(model.children())[:-2])
fix2 = nn.Sequential(*modules2)
z=fix2(img)
# z=z.view(1,196,40,-1)
# z=gap2(relu(conv2(z))).view(-1,768)
print(z.size())
# v_2 = gap2(relu(conv2(fix2(img)))).view(-1,768)
# modules3 = list(model.children())[:-3]
# fix3 = nn.Sequential(*modules3)
# v_3 = gap3(relu(conv3(fix3(img)))).view(-1,768)
# modules4 = list(model.children())[:-4]
# fix4 = nn.Sequential(*modules4)
# v_4 = gap4(relu(conv4(fix4(img)))).view(-1,768)
# modules5 = list(model.children())[:-5]
# fix5 = nn.Sequential(*modules5)
# v_5 = gap5(relu(conv5(fix5(img)))).view(-1,768)
# modules7 = list(model.children())[:-7]
# fix7 = nn.Sequential(*modules7)
# v_7 = gap7(relu(conv7(fix7(img)))).view(-1,768)
# return v_2, v_3, v_4, v_5, v_7
# clsidx = torch.argmax(out)
# print(clsidx.item())

# print(list(model.children()))