import torch
import numpy as np
import cv2

from utils.tree_to_img import tree_to_img_mnist
from data.permutedDataset import MNISTPermutedDataset


dataset = MNISTPermutedDataset()
data0, data1 = dataset[0]
data0 = data0.flatten()
data1 = data1.flatten()
print(data0.shape, data1.shape)

data0 = (data0 + 1.0) / 2.0
data1 = (data1 + 1.0) / 2.0
i = 0
img0, img1 = tree_to_img_mnist(data0, data1)
cv2.imwrite(f"test_imgs/img{i}_0_p.png", img0)
cv2.imwrite(f"test_imgs/img{i}_1_p.png", img1)
