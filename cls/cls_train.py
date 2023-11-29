import torch
from torch import nn
from models import VAE
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from ignite.metrics import Accuracy


class MLP(nn.Module):
    def __init__(self, input_channel, class_num, layer_num, units):
        super(MLP, self).__init__()
        
        layers = []
        for i in range(layer_num):
            if i == 0:
                layers += [nn.Linear(input_channel, units[i]),
                           nn.ReLU()]
            else:
                layers += [nn.Linear(units[i - 1], units[i]),
                           nn.ReLU()]
        
        self.layers = nn.Sequential(*layers)
        
        self.cls_head = nn.Linear(units[-1], class_num)
    
    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        
        x = self.layers(x)
        y_pred = self.cls_head(x)

        return y_pred, x
    
    
def train_loop(dataloader, model, device, optimizer):
    loss_fn = nn.CrossEntropyLoss()
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        y_pred, _ = model(x)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def test_loop(dataloader, model, device):
    metric_fn = Accuracy()
    metric_fn.reset()
    
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        
        y_pred, _ = model(x)
        metric_fn.update((y_pred, y))
    acc = metric_fn.compute()
    print(acc)


if __name__ == '__main__':
    train_dataset = MNIST('data', train=True, transform=ToTensor())
    test_dataset = MNIST('data', train=False, transform=ToTensor())
    
    train_dataloader = DataLoader(train_dataset, 50, shuffle=True)
    test_dataloader = DataLoader(test_dataset, 50, shuffle=False)
    
    device = 'cuda:0'
    
    cls_model = MLP(784, 10, 2, [512, 256]).to(device)
    optimizer = torch.optim.SGD(cls_model.parameters(), lr=0.01, momentum=0.9)
    
    epochs = 10
    for i in range(epochs):
        train_loop(train_dataloader, cls_model, device, optimizer)
        test_loop(test_dataloader, cls_model, device)
    torch.save(cls_model.state_dict(), 'cls_model.pth')
