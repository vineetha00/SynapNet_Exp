# TRACK 2: Streaming / Physiological Data Forecasting / Reconstruction
# Train SynapNet as a regressor on biosignal windows.

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from synapnet import SynapNet

###############################################
# ECGDataset expects CSV with columns like:
# time, sensor1, sensor2, ..., target_clean
###############################################
class ECGDataset(Dataset):
    def __init__(self, path, window=1024, stride=512):
        df = pd.read_csv(path)
        # X: all sensor channels except last
        self.x = df.iloc[:, 1:-1].values.astype(np.float32)
        # y: last column = clean target
        self.y = df.iloc[:, -1].values.astype(np.float32)

        self.window = window
        self.stride = stride
        self.samples = []
        for i in range(0, len(self.x) - window, stride):
            seg_x = self.x[i:i+window]   # (T,C)
            seg_y = self.y[i:i+window]   # (T,)
            self.samples.append((seg_x, seg_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        x = torch.tensor(x, dtype=torch.float32)    # (T,C)
        y = torch.tensor(y, dtype=torch.float32)    # (T,)
        return x, y

###############################################
# SynapNetRegressor:
# - uses SynapNet backbone for contextual features
# - predicts clean 1D signal
###############################################
class SynapNetRegressor(nn.Module):
    def __init__(self, in_dim, embed_dim=64, depth=3):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.backbone = SynapNet(dim=embed_dim,
                                 depth=depth,
                                 vocab_size=1000,
                                 max_len=8192,
                                 num_classes=None)
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B,T,C)
        B, T, C = x.shape
        # dummy token ids just to drive backbone contextual mixing
        idx = torch.randint(low=0, high=999, size=(B, T), device=x.device)
        h, masks = self.backbone(idx)     # h: (B,T,V) because LM head; masks: list[(B,T)]
        # take hidden features BEFORE vocab projection by hijacking embed+pos only?
        # quick hack: reuse last block input by re-embedding raw x
        feats = self.embed(x)             # (B,T,embed_dim)
        # combine feats with h's logits as contextual signal
        feats = feats + h[..., :feats.size(-1)]
        yhat = self.head(feats).squeeze(-1)  # (B,T)
        return yhat, masks

###############################################
# Train loop for regression (MSE)
###############################################
def train_reg(model, loader, device="cuda"):
    mse = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(5):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred, masks = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"[Epoch {epoch+1}] MSE={total/len(loader):.6f}")
    torch.save(model.state_dict(), "synapnet_biosignal.pt")

###############################################
# Visualization helper: plot signal + prediction
###############################################
def visualize_example(model, dataset, device="cuda"):
    model.eval()
    xb, yb = dataset[0]
    xb = xb.unsqueeze(0).to(device)  # (1,T,C)
    yb = yb.unsqueeze(0).to(device)  # (1,T)
    with torch.no_grad():
        pred, masks = model(xb)
    pred = pred.cpu().numpy().flatten()
    gt   = yb.cpu().numpy().flatten()

    plt.figure(figsize=(10,4))
    plt.plot(gt, label="clean target")
    plt.plot(pred, label="model pred")
    plt.title("SynapNet reconstruction")
    plt.legend()
    plt.tight_layout()
    plt.savefig("biosignal_reconstruction.png")

    # visualize salience mask from first block
    plt.figure(figsize=(10,3))
    plt.imshow(masks[0][0].cpu().numpy()[None, :], aspect="auto")
    plt.title("Salience mask (block 0)")
    plt.tight_layout()
    plt.savefig("biosignal_salience.png")

if __name__ == "__main__":
    # You MUST provide your own CSV at data/ecg_sample.csv
    ds = ECGDataset("data/ecg_sample.csv", window=1024, stride=512)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SynapNetRegressor(in_dim=ds.x.shape[1], embed_dim=64, depth=3)
    train_reg(model, dl, device=device)
    visualize_example(model, ds, device=device)
