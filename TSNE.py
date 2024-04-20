import PIL.Image
import torch, torchvision, torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE


class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')


def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


colors_per_class = {
    '0': [254, 202, 87],
    '1': [255, 107, 107],
    '2': [10, 189, 227],
    '3': [255, 159, 243],
    '4': [16, 172, 132],
    '5': [128, 80, 128],
    '6': [87, 101, 116],
    '7': [52, 31, 151],
    '8': [0, 0, 0],
    '9': [100, 100, 255],
}

transform = transforms.Compose([
    GrayscaleToRGB(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transform  # transforms images in tensors
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
features = None
labels = []

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')  # pretrained = True
model.eval()
model.to(device)

for batch in tqdm(train_loader, desc='Running the model inference...'):
    images = batch[0].to(device)
    labels += batch[1]
    with torch.no_grad():
        output = model.forward(images)
    current_features = output.cpu().numpy()  # moves output probabilities tensor to cpu and converts into a np array
    if features is not None:
        features = np.concatenate((features, current_features))
    else:
        features = current_features

tsne = TSNE(n_components=2).fit_transform(features)

tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

fig = plt.figure()
ax = fig.add_subplot(111)  # equivalent to fig.add_subplot(1, 1, 1)

labels = list(map(str, list(map(int, labels))))  # converts a 1-d tensor into a list of strings

for label in colors_per_class:
    indices = [i for i, l in enumerate(labels) if l == label]
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
    color = np.array(colors_per_class[label], dtype=np.float_) / 255
    color = np.expand_dims(color, axis=0)
    ax.scatter(current_tx, current_ty, c=color, label=label)

ax.legend(loc='best')
plt.savefig("output.jpg")
plt.show()






