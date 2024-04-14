"""
The attempt to disect the code for the API usage 
"""

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




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_size = 356 #356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()

    ]
)

    
#image loader convert the image to tensor 
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

#loading the images 
original_img = image_loader(image_name = "content.jpeg")

#the generated image
generated = original_img.clone().requires_grad_(True)


print(original_img)

print(generated)