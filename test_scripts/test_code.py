import sys
sys.path.insert(0, ".")
import torch
import numpy as np
import cv2

from utils.tree_to_img import tree_to_img_mnist
from data.permutedDataset import MNISTPermutedDataset


dataset = MNISTPermutedDataset(aligned_gen=True)

for i in range(10):
    data0, data1, _ = dataset[i]
    data0 = data0.flatten()
    data1 = data1.flatten()
    data0 = (data0 + 1.0) / 2.0
    data1 = (data1 + 1.0) / 2.0
    img0, img1 = tree_to_img_mnist(data0, data1, aligned_gen=True)
    cv2.imwrite(f"test_imgs/img{i}_0_p.png", img0)
    cv2.imwrite(f"test_imgs/img{i}_1_p.png", img1)
