#all the imports 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image

#importing the model 
model = models.vgg19(pretrained=True).features

print(model)

'''
layers 
0,5,10,19,28
'''
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()

        self.chosen_features = ['0','5','10','19','28']
        self.model = models.Vgg19(pretrained= True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)
            
        return features

    
#image loader 
def image_load(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 256 #356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()

    ]
)

#taking the image 
original_img = image_load(image_name = "content.jpeg")
style_img = image_load("style.jpeg")

generated = original_img.clone().requires_grad_(True)

#hyperparamaters

total_steps = 2000
learning_rate = 0.001
alpha = 1 
beta = 0.01

optimizer  = optim.Adam([generated], lr = learning_rate)

for steps in range(total_steps):
    generated_features = model(generated)
    original_img_features = model(original_img)

    style_features = model(style_img)

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
