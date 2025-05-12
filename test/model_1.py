import torch
import torch.nn as nn
from torchvision import models

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #############################
        # Initialize your network
        #############################
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("device=",self.device)
        #model = models.resnet50(pretrained=True)
        self.res = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        # print("self.res!!!!!= ",self.res)
        self.res.fc = nn.Sequential(nn.Flatten(),
                                      nn.Linear(1792, 625),
                                      nn.ReLU(),
                                      nn.Dropout(0.3),
                                      nn.Linear(625, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 8))
        #print("self.res!!!!!= ",self.res.classifier.in_features)
        self.res.to(self.device)
        #print("\n self.res.classifier####= ",self.res)
        
    def forward(self, x):
        
        #############################
        # Implement the forward pass
        #############################
        
        self.res.to(self.device)
        #print("device=",self.device)


        x = self.res(x)
        return x
    
    def save_model(self):
        
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        
        torch.save(self.state_dict(), 'model.pkl')

