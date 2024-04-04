import torch
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
import torchmetrics
from torch import nn , optim
import os

class NNW(nn.Module):
    def __init__(self):
        super(NNW, self).__init__()
        self.resnet = resnet18(weights='IMAGENET1K_V1')
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout=nn.Dropout(0.2)
        self.linear = nn.Linear(1000, 10)
        self.softmax = nn.Softmax(0)

    def forward(self, x):
        x = self.resnet(x)
        # x = self.dropout(x)
        x = self.linear(x)
        return x

class config:
    name='Activity4'
    epochs=10
    lr=1
    batch_size=64
    shuffle=True
    transform=transforms.ToTensor()
    test_frac=0.2
    dirs = ['datasets', 'models']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./datasets', train=False, download=True, transform=transform)
    _,train_dataset =  random_split(train_dataset, [1-test_frac, test_frac])
    _,test_dataset =  random_split(test_dataset, [1-test_frac, test_frac])
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    model = NNW().to(device)
    metric = torchmetrics.classification.Accuracy(
        task="multiclass", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model_name='resnet18_tf_cifar'
    model_file = os.path.join('models', model_name)
    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
    from tqdm import tqdm