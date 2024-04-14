#all the imports 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#list to store the layer value
layer_list_0 = [64,64,'M', 128,128, 'M', 256, 256,256, 'M', 512,512,512, 'M', 512,512,512, 'M']  # the conv layer part 

layer_list_1 = [64,64,'M', 128,128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M', 512,512,512,512, 'M']  # the conv layer part 
#the flatten the layers


class VGG_model(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000):
        super(VGG_model,self).__init__()  #inheritance
        #make the layers now 
        self.in_channels = in_channels
        self.conv_layers = self.create_layers(architecture = layer_list_1)

        self.fcs = nn.Sequential(
            nn.Linear(in_features = 512*7*7, out_features = 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 4096, out_features = 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 4096, out_features = num_classes)

        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x 

    def create_layers(self,architecture):
        layers = []
        in_channels = self.in_channels
        

        #loop iterator  for every architecture
        for i in architecture:
            if type(i) == int:
                out_channels = i

                layers +=[nn.Conv2d(in_channels = in_channels , out_channels= out_channels , 
                kernel_size= (3,3), 
                stride = (1,1),
                padding =(1,1)),
                nn.ReLU()]

                in_channels = i

            elif i == "M":
                layers += [nn.MaxPool2d(kernel_size = (2,2) , stride = (2,2))]

        return nn.Sequential(*layers) 


vgg_model = VGG_model(in_channels= 3 , num_classes= 1000)
print(vgg_model)

test = torch.randn(1,3,224,224)

print(vgg_model(test).shape)