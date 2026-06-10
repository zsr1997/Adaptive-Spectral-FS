import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------
# Exp 3: Binary Proxy (SoftMaskProxyNet)
# ---------------------------------------------------------
class SoftMaskProxyNet(nn.Module):
    def __init__(self, num_features, adaptive=True, fixed_weights=None):
        super().__init__()
        self.adaptive = adaptive
        if self.adaptive:
            self.w_logits = nn.Parameter(torch.tensor([1.1, 1.0, 0.9]))
        else:
            self.register_buffer('fixed_w', torch.tensor(fixed_weights if fixed_weights else [1/3]*3, dtype=torch.float32))
        self.cls_head = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def get_mask(self, m1, m2, m3):
        w = F.softmax(self.w_logits, dim=0) if self.adaptive else self.fixed_w
        return w[0]*m1 + w[1]*m2 + w[2]*m3, w

    def forward(self, m1, m2, m3, x_raw):
        mask, w = self.get_mask(m1, m2, m3)
        return self.cls_head(x_raw * mask), w

def train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, num_features, adaptive=True, fixed_weights=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(TensorDataset(torch.tensor(xtrain, dtype=torch.float32), torch.tensor(ytrain, dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
    model = SoftMaskProxyNet(num_features, adaptive, fixed_weights).to(device)
    opt = torch.optim.Adam([{'params': model.cls_head.parameters()}, {'params': [model.w_logits], 'lr': 0.01}] if adaptive else model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(80):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            m1 = torch.tensor(pop1[np.random.randint(0, len(pop1), bx.shape[0])], dtype=torch.float32).to(device)
            m2 = torch.tensor(pop2[np.random.randint(0, len(pop2), bx.shape[0])], dtype=torch.float32).to(device)
            m3 = torch.tensor(pop3[np.random.randint(0, len(pop3), bx.shape[0])], dtype=torch.float32).to(device)
            logits, _ = model(m1, m2, m3, bx)
            loss = loss_fn(logits, by)
            opt.zero_grad(); loss.backward(); opt.step()
            
    with torch.no_grad():
        final_lambda = F.softmax(model.w_logits, dim=0).cpu().numpy() if adaptive else None
        mask, _ = model.get_mask(torch.tensor(pop1.mean(0), dtype=torch.float32).to(device), torch.tensor(pop2.mean(0), dtype=torch.float32).to(device), torch.tensor(pop3.mean(0), dtype=torch.float32).to(device))
    return mask.cpu().numpy(), final_lambda

# ---------------------------------------------------------
# Exp 1: Multi-Class Proxy (SoftMaskProxyNetMulti)
# ---------------------------------------------------------
class SoftMaskProxyNetMulti(nn.Module):
    def __init__(self, num_features, num_classes, adaptive=True, fixed_weights=None):
        super().__init__()
        self.adaptive = adaptive
        if self.adaptive: self.w_logits = nn.Parameter(torch.tensor([1.1, 1.0, 0.9]))
        else: self.register_buffer('fixed_w', torch.tensor(fixed_weights if fixed_weights else [1/3]*3, dtype=torch.float32))
        self.cls_head = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def get_mask(self, m1, m2, m3):
        w = F.softmax(self.w_logits, dim=0) if self.adaptive else self.fixed_w
        return w[0]*m1 + w[1]*m2 + w[2]*m3, w

    def forward(self, m1, m2, m3, x):
        mask, w = self.get_mask(m1, m2, m3)
        return self.cls_head(x * mask), w

def train_softmask_dnn_proxy_multi(pop1, pop2, pop3, xtrain, ytrain, num_features, num_classes, adaptive=True, fixed_weights=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(TensorDataset(torch.tensor(xtrain, dtype=torch.float32), torch.tensor(ytrain, dtype=torch.long)), batch_size=32, shuffle=True)
    model = SoftMaskProxyNetMulti(num_features, num_classes, adaptive, fixed_weights).to(device)
    opt = torch.optim.Adam([{'params': model.cls_head.parameters()}, {'params': [model.w_logits], 'lr': 0.02}] if adaptive else model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(80):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            m1 = torch.tensor(pop1[np.random.randint(0, len(pop1))], dtype=torch.float32).unsqueeze(0).expand(bx.shape[0], -1).to(device)
            m2 = torch.tensor(pop2[np.random.randint(0, len(pop2))], dtype=torch.float32).unsqueeze(0).expand(bx.shape[0], -1).to(device)
            m3 = torch.tensor(pop3[np.random.randint(0, len(pop3))], dtype=torch.float32).unsqueeze(0).expand(bx.shape[0], -1).to(device)
            logits, _ = model(m1, m2, m3, bx)
            loss = loss_fn(logits, by)
            opt.zero_grad(); loss.backward(); opt.step()
            
    with torch.no_grad():
        final_lambda = F.softmax(model.w_logits, dim=0).cpu().numpy() if adaptive else None
        mask, _ = model.get_mask(torch.tensor(pop1.mean(0), dtype=torch.float32).to(device), torch.tensor(pop2.mean(0), dtype=torch.float32).to(device), torch.tensor(pop3.mean(0), dtype=torch.float32).to(device))
    return mask.cpu().numpy(), final_lambda

# ---------------------------------------------------------
# Exp 2: Dual-Task Proxy (DualProxyMaskNet)
# ---------------------------------------------------------
class DualProxyMaskNet(nn.Module):
    def __init__(self, num_features, adaptive=True, fixed_weights=None):
        super().__init__()
        self.adaptive = adaptive
        if self.adaptive: self.w_logits = nn.Parameter(torch.tensor([1.1, 1.0, 0.9]))
        else: self.register_buffer('fixed_w', torch.tensor(fixed_weights if fixed_weights else [1/3]*3, dtype=torch.float32))
        self.cls_head = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.reg_head = nn.Sequential(nn.Linear(num_features, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def get_mask(self, m1, m2, m3):
        w = F.softmax(self.w_logits, dim=0) if self.adaptive else self.fixed_w
        return w[0]*m1 + w[1]*m2 + w[2]*m3, w

    def forward(self, m1, m2, m3, x):
        mask, w = self.get_mask(m1, m2, m3)
        return self.cls_head(x * mask), self.reg_head(x * mask), w

def train_softmask_dnn_dual_proxy(pop1, pop2, pop3, xtrain, yc_train, yr_train, num_features, adaptive=True, fixed_weights=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yr_norm = (yr_train - np.mean(yr_train)) / (np.std(yr_train) + 1e-8)
    loader = DataLoader(TensorDataset(torch.tensor(xtrain, dtype=torch.float32), torch.tensor(yc_train, dtype=torch.float32).unsqueeze(1), torch.tensor(yr_norm, dtype=torch.float32).unsqueeze(1)), batch_size=32, shuffle=True)
    model = DualProxyMaskNet(num_features, adaptive, fixed_weights).to(device)
    opt = torch.optim.Adam([{'params': model.cls_head.parameters()}, {'params': model.reg_head.parameters()}, {'params': [model.w_logits], 'lr': 0.05}] if adaptive else model.parameters(), lr=1e-3)
    
    model.train()
    for _ in range(80):
        for bx, byc, byr in loader:
            bx, byc, byr = bx.to(device), byc.to(device), byr.to(device)
            m1 = torch.tensor(pop1[np.random.randint(0, len(pop1), bx.shape[0])], dtype=torch.float32).to(device)
            m2 = torch.tensor(pop2[np.random.randint(0, len(pop2), bx.shape[0])], dtype=torch.float32).to(device)
            m3 = torch.tensor(pop3[np.random.randint(0, len(pop3), bx.shape[0])], dtype=torch.float32).to(device)
            c_pred, r_pred, _ = model(m1, m2, m3, bx)
            loss = nn.BCEWithLogitsLoss()(c_pred, byc) + nn.MSELoss()(r_pred, byr)
            opt.zero_grad(); loss.backward(); opt.step()
            
    with torch.no_grad():
        final_lambda = F.softmax(model.w_logits, dim=0).cpu().numpy() if adaptive else None
        mask, _ = model.get_mask(torch.tensor(pop1.mean(0), dtype=torch.float32).to(device), torch.tensor(pop2.mean(0), dtype=torch.float32).to(device), torch.tensor(pop3.mean(0), dtype=torch.float32).to(device))
    return mask.cpu().numpy(), final_lambda