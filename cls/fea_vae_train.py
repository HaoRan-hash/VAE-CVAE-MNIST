import torch
from torch import nn
from models import VAE
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from cls.cls_train import MLP
from tqdm import tqdm


def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x)
        # MSE = nn.functional.mse_loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        # print(f'bce loss={BCE}, kld loss={KLD}')

        return BCE + KLD


def train_loop(dataloader, cls_model, vae, device, optimizer):
    pbar = tqdm(dataloader)
    
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            _, fea = cls_model(x)
        
        # 将数据标准化到[0, 1]
        fea_min, fea_max = fea.min(), fea.max()
        fea = (fea - fea_min) / (fea_max - fea_min)
        
        recon_fea, mean, log_var, _ = vae(fea, y)
        loss = loss_fn(recon_fea, fea, mean, log_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix_str(f'loss={loss:.4f}')
        
        
if __name__ == '__main__':
    train_dataset = MNIST('data', train=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, 50, shuffle=True)
    
    device = 'cuda:0'
    
    cls_model = MLP(784, 10, 2, [512, 256]).to(device)
    cls_model.load_state_dict(torch.load('cls_model.pth', map_location=device))
    cls_model.eval()
    
    vae = VAE([256, 64], 32, [64, 256], conditional=True, num_labels=10).to(device)
    # optimizer = torch.optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    epochs = 20
    for _ in range(epochs):
        train_loop(train_dataloader, cls_model, vae, device, optimizer)
        torch.save(vae.state_dict(), 'fea_vae.pth')
        