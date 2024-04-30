import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from sklearn.manifold import TSNE
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from utilities import GrayscaleToRGB
import matplotlib.pyplot as plt
import numpy as np
import wandb


class ResnetCIFAKE(nn.Module):
    def __init__(self):
        super(ResnetCIFAKE, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def train_model(self, tr_loader, v_loader, t_loader, optim, loss_function, num_epochs, project_name, use_test=False):

        wandb.init(project=project_name, entity='niccolomalgeri')

        if use_test:
            loader = t_loader
            desc = 'testing the model...'
            output_print = "Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}"
            wandb_print_loss = 'test loss'
            wandb_print_acc = 'test accuracy'
        else:
            loader = v_loader
            desc = 'validating the model...'
            output_print = "Epoch [{}/{}], train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}"
            wandb_print_loss = 'validation loss'
            wandb_print_acc = 'validation accuracy'

        if torch.cuda.is_available():
            dev = 'cuda'
        else:
            dev = 'cpu'

        for epoch in range(num_epochs):

            self.model.train()
            train_loss = 0.0
            train_acc = 0

            for batch, labels in tqdm(tr_loader, desc='training the model...'):

                batch = batch.to(dev)
                labels = labels.to(dev)

                optim.zero_grad()
                outputs = self.model(batch)
                predictions = torch.argmax(outputs, 1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optim.step()

                train_loss += loss.item()*batch.size(0)
                train_acc += torch.sum(predictions == labels).double()

            train_loss = train_loss/len(tr_loader.dataset)
            train_acc = train_acc/len(tr_loader.dataset)

            self.model.eval()

            current_loss = 0.0
            current_acc = 0

            with torch.no_grad():
                for batch, labels in tqdm(loader, desc=desc):

                    batch = batch.to(dev)
                    labels = labels.to(dev)

                    outputs = self.model(batch)
                    predictions = torch.argmax(outputs, 1)
                    loss = loss_function(outputs, labels)

                    current_loss += loss*batch.size(0)
                    current_acc += torch.sum(predictions == labels).double()

                current_loss = current_loss/len(loader.dataset)
                current_acc = current_acc/len(loader.dataset)

            wandb.log({'epoch': epoch + 1, 'training loss': train_loss, 'training accuracy': train_acc,
                       wandb_print_loss: current_loss, wandb_print_acc: current_acc})

            print(output_print.format(epoch+1, num_epochs, train_loss, train_acc, current_loss, current_acc))

        wandb.finish()


transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = ResnetCIFAKE()
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model.model.to(device)

train_folder = './data/CIFAKE/train'
test_folder = './data/CIFAKE/test'

train_set = ImageFolder(train_folder, transform=transform)

train_size = int(0.8 * len(train_set))
val_size = len(train_set) - train_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_size, val_size])

test_set = ImageFolder(test_folder, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)

# the new task is pretty different from the initial one, so it's better not to freeze the feature extraction part

'''for param in model.parameters():
    param.requires_grad = False
model.model.fc.requires_grad_(True)'''

optimizer = torch.optim.Adam(model.model.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(model.model.parameters(), lr=0.0001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

model.train_model(train_loader, validation_loader, test_loader, optimizer, loss_fn, num_epochs=10,
                  project_name='CIFAKE_fine_tuning', use_test=False)

torch.save(model.model.state_dict(), './custom_models/CIFAKE_Resnet18')

'''for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.model.parameters(), lr=3e-3)

model.train_model(train_loader, validation_loader, test_loader, optimizer, loss_fn, num_epochs=10, use_test=False)'''
