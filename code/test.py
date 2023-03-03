import albumentations as A
import cv2 as cv
# from PIL import Image
# import numpy as np
# import torch
# from torch import nn
# from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
# import matplotlib.pyplot as plt
#
# from torchvision.transforms.transforms import ToTensor
#
# torch.set_printoptions(profile="full")
# # img = cv.imread("img.jpeg")
# # mask = cv.imread("mask.jpeg")
# #
# # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# # img_tensor = ToTensor()(img).to(torch.float32)
# #
# # mask_tensor = ToTensor()(mask).to(torch.float32)
#
# # img = cv.imread("img0_001.png")
# # img = ToTensor()(Image.open("img0_001.png"))
# #
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

