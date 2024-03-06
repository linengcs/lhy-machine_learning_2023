from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import random_split


def same_seed(seed):
    """Fixes random number generator seeds for reproducibility."""
    # 告知pytorch使用确定性的算法
    torch.backends.cudnn.deterministic = True
    # 在benchmark模式下，CUDNN会在运行时选择最适合当前硬件和输入大小的卷积算法，以提高性能。这里设置成False是为了避免在不同机器上运行结果不同
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    """Split provided data into training and validation sets."""
    valid_set_size = int(valid_ratio * len(data_set))
    train_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds
