from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import re
from PIL import Image

class ChristmasImages(Dataset):
    
    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        self.path=path
        if self.training==True:
            transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            self.dataset = datasets.ImageFolder(self.path, transform=transform)

        else:
            images_unsorted = os.listdir(self.path)
            self.images = self.natural_sort(images_unsorted)
            self.test_path = self.path
            self.test_transforms = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224), transforms.ToTensor()])

        # If training == True, path contains subfolders
        # containing images of the corresponding classes
        # If training == False, path directly contains
        # the test images for testing the classifier
    def natural_sort(self,l):
        convert = lambda text: int (text) if text.isdigit() else text.lower()
        alphanum_key = lambda key : [convert(c) for c in re.split('([0-9]+)',key)]
        return sorted(l,key = alphanum_key)
    
    def __len__(self):
        if self.training:
            return len(self.dataset)        
        else:
            return len(self.images)
    
    def __getitem__(self, index):
        if self.training:
            return self.dataset[index] 
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        
        else:
            images_location = os.path.join(self.test_path, self.images[index])
            image = Image.open(images_location).convert('RGB')
            image = self.test_transforms(image)
            return (image,)
