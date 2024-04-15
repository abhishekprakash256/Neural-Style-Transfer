"""
The model is  a dummy model for the traninhg of the pytrorch data 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(in_features = 4096, out_features = 1000)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#print(net)

#make the random tensor 

input_height = 32  # Example height
input_width = 32   # Example width
batch_size = 16    # Example batch size

# Generate a random tensor with the desired size
input_image1 = torch.randn(batch_size, 3, input_height, input_width)

input_image2 = torch.randn(batch_size, 3, input_height, input_width)

generated = torch.randn(batch_size, 3, input_height, input_width)


total_steps = 2000
learning_rate = 0.001
alpha = 1 
beta = 0.01


for steps in range(total_steps):
    generated_features = net(generated)
    original_img_features = net(input_image1)

    style_features = net(input_image2)

    style_loss = original_loss = 0

    for gen_feature, origL_feature , style_feature in zip(
        generated_features, original_img_features,style_features
    ):

        channel , height , width = gen_feature.shape

        original_loss += torch.mean((gen_feature - origL_feature) ** 2)

        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel,  height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G-A)**2)

    total_loss = alpha*original_loss + beta * style_loss

    optimizer.zero_grad()

    total_loss.backward()
    optimizer.step()

    if steps %200 == 0:
        print(total_loss)
        save_image(generated, "generated.png")
