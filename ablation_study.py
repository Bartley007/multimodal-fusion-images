# -*- coding: utf-8 -*-
"""
Ablation Study for Attention Fusion
- Image + Text
- Image + Time
- Text + Time
- Full (Image + Text + Time)
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Reuse functions from attention_fusion_fit.py
from attention_fusion_fit import (
    load_sources, set_seed, TrioDataset,
    SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, TEST_SIZE, THRESH
)

OUT_DIR = "./ablation_out"; os.makedirs(OUT_DIR, exist_ok=True)

class FlexibleAttnFusion(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6, hidden_size: int = 8):
        super().__init__()
        self.eps = eps
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, input_dim)

    def forward(self, p):
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        z = torch.log(p) - torch.log(1.0 - p)
        s = self.fc2(self.act(self.fc1(z)))
        a = torch.softmax(s, dim=-1)
        fused_logit = torch.sum(a * z, dim=-1, keepdim=True)
        fused_prob = torch.sigmoid(fused_logit)
        return fused_prob, a

def train_eval_ablation(merged, feature_cols, device, experiment_name):
    print(f"\n>>> Running Ablation: {experiment_name} (Features: {feature_cols})")
    X = merged[feature_cols].to_numpy()
    y = merged["binary_true"].to_numpy()
    ids = merged["id"].to_numpy()

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    train_ds = TrioDataset(X_train, y_train)
    test_ds  = TrioDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor((neg / max(1, pos)) if pos > 0 else 1.0, dtype=torch.float32, device=device)

    model = FlexibleAttnFusion(input_dim=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(loader, train_mode=False):
        if train_mode: model.train()
        else: model.eval()
        losses, probs_all, labels_all = [], [], []
        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                prob, _ = model(xb)
                prob = prob.squeeze(1)
                loss_pos = - yb * torch.log(torch.clamp(prob, 1e-6, 1-1e-6))
                loss_neg = - (1 - yb) * torch.log(torch.clamp(1 - prob, 1e-6, 1-1e-6))
                loss = (pos_weight * loss_pos + loss_neg).mean()

                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())
                probs_all.append(prob.detach().cpu().numpy())
                labels_all.append(yb.detach().cpu().numpy())

        probs = np.concatenate(probs_all)
        labels = np.concatenate(labels_all)
        preds = (probs >= THRESH).astype(int)
        acc = accuracy_score(labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return f1, acc

    set_seed(SEED)
    for ep in range(1, EPOCHS+1):
        run_epoch(train_loader, train_mode=True)
    
    f1, acc = run_epoch(test_loader, train_mode=False)
    
    # 获取测试集的预测概率用于 ROC 曲线
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device); yb = yb.to(device)
            prob, _ = model(xb)
            probs_all.append(prob.squeeze(1).cpu().numpy())
            labels_all.append(yb.cpu().numpy())
    
    test_probs = np.concatenate(probs_all)
    test_labels = np.concatenate(labels_all)
    
    print(f"Result: Macro-F1={f1:.4f}, Acc={acc:.4f}")
    return {"experiment": experiment_name, "f1": f1, "acc": acc, "probs": test_probs.tolist(), "labels": test_labels.tolist()}

def main():
    set_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    merged, _ = load_sources()

    experiments = [
        ("Image + Text", ["img_prob", "txt_prob"]),
        ("Image + Time", ["img_prob", "time_prob"]),
        ("Text + Time",  ["txt_prob", "time_prob"]),
        ("Full (Image+Text+Time)", ["img_prob", "txt_prob", "time_prob"])
    ]

    results = []
    for name, cols in experiments:
        res = train_eval_ablation(merged, cols, DEVICE, name)
        results.append(res)

    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False, encoding="utf-8-sig")
    
    # 保存详细的预测结果用于 ROC 曲线绘制
    with open(os.path.join(OUT_DIR, "ablation_probs.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results))
    plt.bar(x - 0.2, [r['f1'] for r in results], 0.4, label='Macro-F1')
    plt.bar(x + 0.2, [r['acc'] for r in results], 0.4, label='Accuracy')
    plt.xticks(x, [r['experiment'] for r in results])
    plt.ylabel('Score')
    plt.title('Ablation Study: Dual-Modality vs Full-Modality Fusion')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ablation_comparison.png"), dpi=160)
    plt.close()

    print(f"\n[DONE] Ablation study finished. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
