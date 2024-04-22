import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from utilities import GrayscaleToRGB
import matplotlib.pyplot as plt
import numpy as np

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


class MnistResnet18(nn.Module):
    def __init__(self):
        super(MnistResnet18, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        #self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # first layer now take 1 color channel
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)  # last layer now outputs 10 classes instead of 1000

    def train_model(self, tr_loader, t_loader, loss_function, opt, num_epochs):

        features = None
        lab = []

        for epoch in range(num_epochs):

            if torch.cuda.is_available():
                dev = 'cuda'
            else:
                dev = 'cpu'

            self.model.train()
            current_loss = 0.0
            current_acc = 0

            for inputs, labels in tqdm(tr_loader, desc="training the model..."):
                inputs = inputs.to(dev)
                labels = labels.to(dev)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, 1)
                loss = loss_function(outputs, labels)
                loss.backward()
                opt.step()

                current_loss += loss.item()*inputs.size(0)
                current_acc += torch.sum((predictions == labels)).double()

            tr_loss = current_loss / len(tr_loader)
            tr_acc = current_acc / len(tr_loader)

            self.model.eval()

            current_loss = 0.0
            current_acc = 0

            with torch.no_grad():
                for inputs, labels in tqdm(t_loader, desc="testing the model..."):
                    inputs = inputs.to(dev)
                    labels = labels.to(dev)

                    outputs = self.model(inputs)
                    if epoch == num_epochs-1:
                        lab += labels
                        current_features = outputs.cpu().numpy()
                        if features is None:
                            features = current_features
                        else:
                            features = np.concatenate((features, current_features))
                    predictions = torch.argmax(outputs, 1)
                    loss = loss_function(outputs, labels)

                    current_loss += loss.item()*inputs.size(0)  # cross-entropy computes a mean wrt the batch, therefore we multiply by batch_size
                    current_acc += torch.sum((predictions == labels)).double()

            t_loss = current_loss / len(t_loader)
            t_acc = current_acc / len(t_loader)

            print("Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}"
                  .format(epoch+1, num_epochs, tr_loss, tr_acc, t_loss, t_acc))

        tsne = TSNE(n_components=2).fit_transform(features)

        tx = tsne[:, 0]
        ty = tsne[:, 1]
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        lab = list(map(str, list(map(int, lab))))

        for label in colors_per_class:
            indices = [i for i, l in enumerate(lab) if l == label]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            color = np.array(colors_per_class[label], dtype=np.float_) / 255
            color = np.expand_dims(color, axis=0)
            ax.scatter(current_tx, current_ty, c=color, label=label)

        ax.legend(loc='best')
        plt.savefig("./plots/outputFineTuning.jpg")
        plt.show()


transform = transforms.Compose([
    GrayscaleToRGB(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transform
)

#TODO: try using torch.utils.data.random_split() to extract a validation set

test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)

model = MnistResnet18()

# freeze every layer except for the last one
for param in model.parameters():
    param.requires_grad = False
model.model.fc.requires_grad_(True)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.model.fc.parameters(), lr=0.001, momentum=0.9)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model.model.to(device)

# fine tune last layer
model.train_model(train_loader, test_loader, loss_fn, optimizer, num_epochs=5)

# unfreeze the original layers and repeat the training
for param in model.parameters():
    param.requires_grad = True

model.train_model(train_loader, test_loader, loss_fn, optimizer, num_epochs=10)





