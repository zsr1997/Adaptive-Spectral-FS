# core/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class SoftMaskProxyNet(nn.Module):
    def __init__(self, num_features, adaptive=True, fixed_weights=None):
        super().__init__()
        self.adaptive = adaptive
        if self.adaptive:
            self.w_logits = nn.Parameter(torch.tensor([1.1, 1.0, 0.9]))
        else:
            if fixed_weights is None:
                fixed_weights = [1/3, 1/3, 1/3]
            self.register_buffer('fixed_w', torch.tensor(fixed_weights, dtype=torch.float32))
            
        self.cls_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def get_mask(self, m1, m2, m3):
        w = F.softmax(self.w_logits, dim=0) if self.adaptive else self.fixed_w
        fused_mask = w[0]*m1 + w[1]*m2 + w[2]*m3
        return fused_mask, w

    def forward(self, m1, m2, m3, x_raw):
        fused_mask, w = self.get_mask(m1, m2, m3)
        masked_x = x_raw * fused_mask
        cls_logits = self.cls_head(masked_x)
        return cls_logits, w

def train_softmask_dnn_proxy(pop1, pop2, pop3, xtrain, ytrain, num_features, adaptive=True, fixed_weights=None, epochs=80, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    x_tensor = torch.tensor(xtrain, dtype=torch.float32)
    y_tensor = torch.tensor(ytrain, dtype=torch.float32).unsqueeze(1)
    dataloader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=32, shuffle=True)
    
    model = SoftMaskProxyNet(num_features, adaptive=adaptive, fixed_weights=fixed_weights).to(device)
    
    if adaptive:
        optimizer = torch.optim.Adam([
            {'params': model.cls_head.parameters(), 'lr': lr},
            {'params': [model.w_logits], 'lr': 0.01}  
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    
    for epoch in range(epochs):
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            idx1 = np.random.randint(0, len(pop1), bx.shape[0])
            idx2 = np.random.randint(0, len(pop2), bx.shape[0])
            idx3 = np.random.randint(0, len(pop3), bx.shape[0])
            
            m1_batch = torch.tensor(pop1[idx1], dtype=torch.float32).to(device)
            m2_batch = torch.tensor(pop2[idx2], dtype=torch.float32).to(device)
            m3_batch = torch.tensor(pop3[idx3], dtype=torch.float32).to(device)
            
            cls_logits, current_w = model(m1_batch, m2_batch, m3_batch, bx)
            loss = loss_fn(cls_logits, by)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
    final_lambda_val = None
    if adaptive:
        with torch.no_grad():
            final_lambda_val = F.softmax(model.w_logits, dim=0).cpu().numpy()
            
    model.eval()
    with torch.no_grad():
        m1_mean = torch.tensor(pop1.mean(0), dtype=torch.float32).to(device).unsqueeze(0)
        m2_mean = torch.tensor(pop2.mean(0), dtype=torch.float32).to(device).unsqueeze(0)
        m3_mean = torch.tensor(pop3.mean(0), dtype=torch.float32).to(device).unsqueeze(0)
        final_mask, _ = model.get_mask(m1_mean, m2_mean, m3_mean)
        
    return final_mask.squeeze(0).cpu().numpy(), final_lambda_val