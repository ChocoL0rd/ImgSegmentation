import albumentations as A
import cv2 as cv
from PIL import Image
# import numpy as np
import torch
# from torch import nn
# from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
# import matplotlib.pyplot as plt
#
from torchvision.transforms.transforms import ToTensor
from torch.nn import functional as F

# torch.set_printoptions(profile="full")
# img = cv.imread("img.jpeg")
# mask = cv.imread("mask.jpeg")
#
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img_tensor = ToTensor()(img).to(torch.float32)
#
# mask_tensor = ToTensor()(mask).to(torch.float32)
#
# img = ToTensor()(Image.open("../data/imgs/img0_001.jpeg"))
#
# print(img)

# a = torch.Tensor([1, 2, 4])
# b = a > 3
# print(a*b)

# # new_img = img[:-1]
# # mask = img[-1:]
# # print(new_img)
# # print(new_img.shape)
# # print(mask)
# # print(mask.shape)
#
#
# # plt.imshow(img_tensor.reshape(3, 330, 330)[0])
# # plt.show()
# # transform = A.Compose([
# #     A.RandomCrop(width=256, height=256),
# #     A.HorizontalFlip(p=0.5),
# #     A.RandomBrightnessContrast(p=0.2),
# #     ChannelDropout()
# # ])
# # #
# #
# # trfmed = transform(image=img, mask=mask)
# # trfmed_image = trfmed["image"]
# # trfmed_mask = trfmed["mask"]
# # print(trfmed_image.shape)
# # print(trfmed_mask.shape)
# # # print(type(A.RandomCrop(width=256, height=256)))
# #
# # print(trfmed_image)
#
# # # img = Image.open("img0_001.png")
# # # print(img.mode)
# #
# # img = cv.imread("img0_001.png")
# #
# # # im.show()
# #
# # mask = ToTensor()(img[-1])
# # print(mask.shape)
# # m = nn.Sigmoid()
# # loss = nn.BCELoss()
# # input = torch.randn([1, 1, 330, 330], requires_grad=True)
# # target = torch.empty(1, 1, 330, 330).random_(2)
# #
# # new_input = input.reshape(1, 1, -1)
# # new_target = target.reshape(1, 1, -1)
# #
# # output = loss(m(input), target)
# #
# # print(output)
# # print(loss(m(new_input), new_target))
#
# a = torch.randn([5, 1, 10, 10])
#
# print(a.sum([-1, -2]).shape)

# a, b = [[1,2], [4, 5], [4, 5]]
#
# print(a, b)
#
# from tools.train_tools import ImgMaskSet
#
# dataset = ImgMaskSet(img_dir_path="../data/imgs", mask_dir_path="../data/segmented_imgs",
#            transforms=None, log_name="just",
#            device=torch.device("cpu"), img_list=None)
#
# for img, mask in dataset:
#     print(img.min(), img.max(), mask.min(), mask.max())

# a = torch.rand([5, 3, 11, 11])
# b = F.interpolate(a, size=[21, 21], mode='bilinear')
# print(b.shape)

# for i in range(0, 60, 2):
#     print("{\n"
#           f"name: {'dice'}, \n"
#           f"threshold: {i/100}\n"
#           "},")
#
#
# for i in range(0, 60, 2):
#     print("{\n"
#           f"name: {'jaccard'}, \n"
#           f"threshold: {i/100}\n"
#           "},")
#
# print([i/100 for i in range(0, 60, 2)])

# from torchvision import models
# resnet = models.resnet101()
# # print(resnet.state_dict())
# print(resnet)


a = []
a.append({"djfd": 314})
print(a)