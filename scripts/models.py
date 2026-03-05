import torch 
import torch.nn as nn 

class MashupCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.filters = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classfy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self,x):
        x = self.filters(x)
        x = self.avg_pool(x)
        y = self.classfy(x)
        return y 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
def get_cnn():
    model = MashupCNN(num_classes=10).to(device)
    return model 
