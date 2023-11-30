import torch
from torch.utils.data import DataLoader
from dataset import S3dis
from data_aug import *
from pointmeta import PointMeta
import sys
sys.path.append('/mnt/Disk16T/chenhr/VAE-CVAE-MNIST')
from models import VAE
from tqdm import tqdm


def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x)
        # MSE = nn.functional.mse_loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        # print(f'bce loss={BCE}, kld loss={KLD}')

        return BCE + KLD


def train_loop(dataloader, cls_model, vae, device, optimizer):
    pbar = tqdm(dataloader)
    
    for pos, color, y in pbar:
        pos = pos.to(device)
        color = color.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            _, fea = cls_model(pos, color)
        fea, y = fea.squeeze(dim=0), y.squeeze(dim=0)
        
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
    train_aug = Compose([ColorContrast(p=0.2),
                         PointCloudScaling(0.9, 1.1),
                         PointCloudFloorCentering(),
                         PointCloudRotation_Z(1.0, False),
                         PointCloudJitter(0.005, 0.02),
                         ColorDrop(p=0.2),
                         ColorNormalize()])
    train_dataset = S3dis('/mnt/Disk16T/chenhr/threed_data/data/processed_s3dis', split='train', loop=30, npoints=24000, transforms=train_aug)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    
    device = 'cuda:0'

    seg_model = PointMeta(13, 4, 32, [4, 8, 4, 4]).to(device)
    seg_model.load_state_dict(torch.load('/mnt/Disk16T/chenhr/VAE-CVAE-MNIST/seg/pointmeta_seg_0001lr.pth', map_location=device)['model_state_dict'])
    seg_model.eval()
    
    vae = VAE([64, 32], 16, [32, 64], conditional=True, num_labels=13).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    epochs = 10
    for _ in range(epochs):
        train_loop(train_dataloader, seg_model, vae, device, optimizer)
        torch.save(vae.state_dict(), 'fea_vae.pth')
    