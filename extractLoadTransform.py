import PIL.Image
import torch, torchvision, torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from utilities import GrayscaleToRGB

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([GrayscaleToRGB(), transforms.ToTensor()]),  # transforms images in tensors
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)  # wrapper for training data

print(len(train_set))  # number of images in the training set
print(train_set.targets)  # list of training labels
print(train_set.targets.bincount())  # number of images relative to each label (i.e. the training labels' distribution)

sample = train_set[0]  # the same as sample = next(iter(train_set))

# sequence/list unpacking:
image = sample[0]
label = sample[1]

plt.imshow(image[0])
plt.show()
print("label:", label)

#batch = next(iter(train_loader))

#batch_images = batch[0]
#batch_labels = batch[1]
#grid = torchvision.utils.make_grid(batch_images, nrow=10)  # nrow = number of images per row
#plt.figure(figsize=(15, 15))  # specifies the inches of the displayed images
#plt.imshow(np.transpose(grid, (1, 2, 0)))  # shows the grid and changes the axis from (C, W, H) to (W, H, C)
#plt.show()

