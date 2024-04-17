import torch, torchvision, torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from PIL import Image
import requests

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
img = Image.open(requests.get(url, stream=True).raw)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = transform(img)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(1, 3, 1)
plt.title("color channel 1")
plt.axis('off')
plt.imshow(input_tensor[0])
fig.add_subplot(1, 3, 2)
plt.title("color channel 2")
plt.axis('off')
plt.imshow(input_tensor[1])
fig.add_subplot(1, 3, 3)
plt.title("color channel 3")
plt.axis('off')
plt.imshow(input_tensor[2])
#plt.imshow(input_tensor[2])
#plt.axis('off')
plt.show()

input_batch = input_tensor.unsqueeze(dim=0)  # tensor of shape [1, 3, 224, 224]
print(input_batch.shape)

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# print(model)
model.eval()

with torch.no_grad():  # disables the weights' update
    output = model(input_batch)  # inference on the model

probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

