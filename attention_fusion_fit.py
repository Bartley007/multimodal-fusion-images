# -*- coding: utf-8 -*-
"""
Attention 融合三路拟合值（自动学习权重，鲁棒列名/ID 识别）
输入(按 id 对齐，脚本会自动规范化):
  1) 图像(按ID聚合)：predictions_id_aggregated.csv    -> 需要: id(或可推断), 概率列, 真值列(或 raw_label)
  2) 文本/表格：      predictions_binary_with_fit.csv  -> 需要: id(或可推断), 概率列
  3) 时间：           predictions_with_time_only.csv   -> 需要: id(或可推断), 概率列

输出：
  ./fusion_out/result.csv
  ./fusion_out/cm_attention.png
  ./fusion_out/metrics.txt
  ./fusion_out/shap_summary.png        # 新增：SHAP beeswarm（若 shap 可用）
  ./fusion_out/shap_bar.png            # 新增：SHAP 特征重要性条形图
"""

import os
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== 路径与配置 =====================
IMAGE_FILE_CAND = [
    "./predictions_id_aggregated.csv",
    "./vit_finetune_out/predictions_id_aggregated.csv"
]
TEXT_FILE_CAND  = [
    "./predictions_binary_with_fit.csv",
    "./predictions_image_with_fit.csv",
    "./text_out/predictions_binary_with_fit.csv"
]
TIME_FILE_CAND  = [
    "./predictions_with_time_only.csv",
    "./time_fit_no_image/predictions_with_time_only.csv",
    "./time_fit/predictions_with_time.csv"
]

OUT_DIR = "./fusion_out"; os.makedirs(OUT_DIR, exist_ok=True)
OUT_RESULT = os.path.join(OUT_DIR, "result.csv")
OUT_CM     = os.path.join(OUT_DIR, "cm_attention.png")
OUT_METRIC = os.path.join(OUT_DIR, "metrics.txt")
OUT_SHAP_SUM = os.path.join(OUT_DIR, "shap_summary.png")
OUT_SHAP_BAR = os.path.join(OUT_DIR, "shap_bar.png")

SEED = int(os.environ.get("SEED", "42"))
BATCH_SIZE = int(os.environ.get("FUSION_BATCH_SIZE", "128"))
EPOCHS = int(os.environ.get("FUSION_EPOCHS", "50"))
PATIENCE = int(os.environ.get("FUSION_PATIENCE", "6"))
LR = float(os.environ.get("FUSION_LR", "5e-3"))
WEIGHT_DECAY = float(os.environ.get("FUSION_WEIGHT_DECAY", "1e-4"))
VAL_SIZE = float(os.environ.get("FUSION_VAL_SIZE", "0"))
TEST_SIZE = float(os.environ.get("TEST_RATIO", os.environ.get("FUSION_TEST_RATIO", "0.2")))
NUM_WORKERS = 0
THRESH = 0.5

# ===================== 小工具 =====================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def pick_first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def sigmoid_np(x):
    return 1. / (1. + np.exp(-x))

def is_prob_series(s: pd.Series):
    try:
        v = pd.to_numeric(s, errors="coerce")
        frac = ((v>=0) & (v<=1)).mean()
        return frac > 0.95
    except Exception:
        return False

def try_extract_id_from_filename(x: str):
    """
    从文件名/路径中回退提取 id：
      - "123_0.jpg" -> "123"
      - "123.jpg"   -> "123"
      - "/path/123_1.png" -> "123"
    """
    if not isinstance(x, str):
        return None
    base = os.path.basename(x)
    name, _ = os.path.splitext(base)
    if "_" in name:
        return name.split("_")[0]
    return name

def coerce_id_column(df: pd.DataFrame, file_tag: str):
    """
    规范化 df，使其有一列 'id'（字符串）。
    识别顺序：
      1) 直接命中常见列名：id/ID/Id/编号/样本id/uid/user_id
      2) 若存在 image_name/filename/path，从文件名回退提取 id
      3) 若首列未命名或叫 Unnamed: 0，且看起来不像概率列，则当作 id
      4) 如果有任何一列是字符串、唯一值丰富，且不是概率列，优先选之
    """
    orig_cols = list(df.columns)
    lower_map = {c.lower(): c for c in df.columns}

    id_alias = ["id","ID","Id","编号","样本id","uid","user_id"]
    for a in id_alias:
        if a in df.columns:
            out = df.rename(columns={a: "id"}).copy()
            out["id"] = out["id"].astype(str)
            print(f"[INFO]({file_tag}) 使用列 '{a}' 作为 id")
            return out

    # 从文件名回退
    for name_col in ["image_name","filename","file_name","path","filepath","file_path"]:
        if name_col in df.columns:
            tmp_id = df[name_col].apply(try_extract_id_from_filename)
            if tmp_id.notnull().any():
                out = df.copy()
                out.insert(0, "id", tmp_id.astype(str))
                print(f"[INFO]({file_tag}) 由列 '{name_col}' 回退提取 id")
                return out

    # 试首列
    first_col = df.columns[0]
    if (first_col.lower().startswith("unnamed")
        or first_col.lower() in ["index","序号"]
        or not is_prob_series(df[first_col])):
        out = df.copy()
        out.insert(0, "id", df[first_col].astype(str))
        print(f"[INFO]({file_tag}) 使用首列 '{first_col}' 作为 id（推断）")
        return out

    # 选择一个“更像 id 的字符串列”
    for c in df.columns:
        s = df[c]
        if s.dtype == object:
            uniq = s.nunique(dropna=True)
            if uniq > max(10, len(df)*0.5) and not is_prob_series(s):
                out = df.copy()
                out.insert(0, "id", s.astype(str))
                print(f"[INFO]({file_tag}) 使用字符串列 '{c}' 作为 id（启发式）")
                return out

    # 实在不行，就把首列当 id（兜底）
    out = df.copy()
    out.insert(0, "id", df[first_col].astype(str))
    print(f"[WARN]({file_tag}) 无法可靠识别 id，兜底使用首列 '{first_col}' 作为 id")
    return out

def coerce_prob_from_cols(df: pd.DataFrame, file_tag: str,
                          prob_candidates=None, logit_candidates=None):
    """
    自动提取“概率列”：
      1) 在 prob_candidates 中找到（0~1）
      2) 在 logit_candidates 中寻找，sigmoid(logit)
      3) 任何包含 "prob" 的列（0~1）
      4) 任何包含 "logit" 的列（sigmoid）
      5) 退而求其次：扫描所有列，挑一个看起来像概率的数值列
    返回：(Series 概率列, 实际使用的列名, 是否为 logit 转换)
    """
    if prob_candidates is None:
        prob_candidates = []
    if logit_candidates is None:
        logit_candidates = []

    # 1) 指定候选
    for c in prob_candidates:
        if c in df.columns and is_prob_series(df[c]):
            print(f"[INFO]({file_tag}) 概率列使用 '{c}'")
            return pd.to_numeric(df[c], errors="coerce"), c, False

    for c in logit_candidates:
        if c in df.columns:
            print(f"[INFO]({file_tag}) 由 logit 列 '{c}' sigmoid 转概率")
            return sigmoid_np(pd.to_numeric(df[c], errors="coerce")), c, True

    # 2) 包含 prob 的列
    for c in df.columns:
        if "prob" in c.lower() and is_prob_series(df[c]):
            print(f"[INFO]({file_tag}) 自动发现概率列 '{c}'")
            return pd.to_numeric(df[c], errors="coerce"), c, False

    # 3) 包含 logit 的列
    for c in df.columns:
        if "logit" in c.lower():
            print(f"[INFO]({file_tag}) 自动发现 logit 列 '{c}' 并 sigmoid 转概率")
            return sigmoid_np(pd.to_numeric(df[c], errors="coerce")), c, True

    # 4) 扫描所有数值列，挑选最像概率的
    num_cols = []
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notnull().any():
            frac = ((v>=0) & (v<=1)).mean()
            num_cols.append((frac, c))
    num_cols.sort(reverse=True)
    if num_cols and num_cols[0][0] > 0.8:
        c = num_cols[0][1]
        print(f"[INFO]({file_tag}) 兜底选择数值列 '{c}' 作为概率（>80%在[0,1]）")
        return pd.to_numeric(df[c], errors="coerce"), c, False

    raise ValueError(f"[{file_tag}] 未在列中找到概率/可转换的 logit 列。可用列: {list(df.columns)}")

def find_label_column(df: pd.DataFrame, file_tag: str):
    """
    自动识别真值列名。优先：binary_true > label/y/target/true/gt > raw_label(二值化)
    返回：(Series 真值(0/1), 使用的列名, 是否从 raw_label 映射)
    """
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["binary_true", "label", "y", "target", "true", "gt"]:
        if key in lower_map:
            col = lower_map[key]
            ser = pd.to_numeric(df[col], errors="coerce")
            if ser.dropna().isin([0,1]).mean() > 0.95:
                print(f"[INFO]({file_tag}) 真值列使用 '{col}'")
                return ser, col, False

    # raw_label 二值化（-1→0，{0,1}→1）
    if "raw_label" in df.columns:
        def binarize(x):
            s = str(x).strip()
            if s == "-1": return 0
            if s in {"0","1",0,1}: return 1
            return np.nan
        ser = df["raw_label"].apply(binarize)
        if ser.notnull().any():
            print(f"[INFO]({file_tag}) 由 'raw_label' 二值化得到真值")
            return ser.astype("float"), "raw_label->binary_true", True

    # 宽松找：包含 "true" 或 "label"
    for c in df.columns:
        lc = c.lower()
        if "true" in lc or "label" in lc:
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.dropna().isin([0,1]).mean() > 0.8:
                print(f"[INFO]({file_tag}) 宽松匹配真值列 '{c}'")
                return ser, c, False

    raise ValueError(f"[{file_tag}] 未找到真值列（binary_true/label/y/target/true/gt/raw_label）。可用列: {list(df.columns)}")

# ===================== 数据加载与对齐 =====================
def load_sources():
    img_path  = pick_first_existing(IMAGE_FILE_CAND)
    txt_path  = pick_first_existing(TEXT_FILE_CAND)
    time_path = pick_first_existing(TIME_FILE_CAND)
    if not img_path:
        raise FileNotFoundError(f"未找到图像聚合结果文件。尝试路径: {IMAGE_FILE_CAND}")
    if not txt_path:
        raise FileNotFoundError(f"未找到文本/表格结果文件。尝试路径: {TEXT_FILE_CAND}")
    if not time_path:
        raise FileNotFoundError(f"未找到时间结果文件。尝试路径: {TIME_FILE_CAND}")

    def _read_csv_fallback(p):
        for _enc in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
            try:
                return pd.read_csv(p, encoding=_enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f"CSV编码不支持，请使用 UTF-8/UTF-8-SIG 或 GBK 编码保存: {p}")

    df_img  = _read_csv_fallback(img_path)
    df_txt  = _read_csv_fallback(txt_path)
    df_time = _read_csv_fallback(time_path)

    # 统一 id
    df_img  = coerce_id_column(df_img,  "IMAGE")
    df_txt  = coerce_id_column(df_txt,  "TEXT")
    df_time = coerce_id_column(df_time, "TIME")

    # 概率列 + 真值列
    img_prob, img_prob_col, _ = coerce_prob_from_cols(
        df_img, "IMAGE",
        prob_candidates=["fit_prob_pos", "img_prob", "prob", "p_pos", "pred_prob"],
        logit_candidates=["logit", "img_logit"]
    )
    y_true_ser, y_true_col, y_from_raw = find_label_column(df_img, "IMAGE")

    df_img_std = pd.DataFrame({
        "id": df_img["id"].astype(str),
        "binary_true": pd.to_numeric(y_true_ser, errors="coerce"),
        "img_prob": pd.to_numeric(img_prob, errors="coerce")
    }).dropna(subset=["img_prob"]).reset_index(drop=True)

    txt_prob, txt_prob_col, _ = coerce_prob_from_cols(
        df_txt, "TEXT",
        prob_candidates=["fit_prob_pos", "txt_prob", "prob", "p_pos", "pred_prob"],
        logit_candidates=["logit", "txt_logit"]
    )
    df_txt_std = pd.DataFrame({
        "id": df_txt["id"].astype(str),
        "txt_prob": pd.to_numeric(txt_prob, errors="coerce")
    }).dropna(subset=["txt_prob"]).reset_index(drop=True)

    time_prob, time_prob_col, _ = coerce_prob_from_cols(
        df_time, "TIME",
        prob_candidates=["time_fit_prob_pos", "time_prob", "prob", "p_pos", "pred_prob"],
        logit_candidates=["time_logit", "logit"]
    )
    df_time_std = pd.DataFrame({
        "id": df_time["id"].astype(str),
        "time_prob": pd.to_numeric(time_prob, errors="coerce")
    }).dropna(subset=["time_prob"]).reset_index(drop=True)

    # 合并（仅保留三路都存在的 id）
    merged = df_img_std.merge(df_txt_std, on="id", how="inner").merge(df_time_std, on="id", how="inner")
    merged = merged.dropna(subset=["binary_true", "img_prob", "txt_prob", "time_prob"]).reset_index(drop=True)
    merged["binary_true"] = merged["binary_true"].astype(int)

    if merged.empty:
        raise RuntimeError("三路结果合并后为空，请检查 id 是否一致以及概率列是否正确。")

    print("\n[INFO] 列使用情况：")
    print(f"  IMAGE: id='id'  prob='{img_prob_col}'  y='{y_true_col}'  (from_raw={y_from_raw})  -> {len(df_img_std)} rows")
    print(f"  TEXT : id='id'  prob='{txt_prob_col}'                       -> {len(df_txt_std)} rows")
    print(f"  TIME : id='id'  prob='{time_prob_col}'                      -> {len(df_time_std)} rows\n")

    return merged, (img_path, txt_path, time_path)

# ===================== 数据集与模型 =====================
class TrioDataset(Dataset):
    def __init__(self, probs_np, labels_np):
        self.X = probs_np.astype(np.float32)  # [N,3]
        self.y = labels_np.astype(np.float32) # [N]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class AttnFusion(nn.Module):
    """
    输入：p ∈ R^3（概率）
    1) z = logit(p)
    2) a = softmax(MLP(z))
    3) fused_logit = Σ a_i * z_i, fused_prob = sigmoid(fused_logit)
    """
    def __init__(self, eps: float = 1e-6, hidden_size: int = 8):
        super().__init__()
        self.eps = eps
        self.fc1 = nn.Linear(3, hidden_size)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, p):  # p: [B,3]
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        z = torch.log(p) - torch.log(1.0 - p)  # logit
        s = self.fc2(self.act(self.fc1(z)))    # [B,3]
        a = torch.softmax(s, dim=-1)           # 注意力
        fused_logit = torch.sum(a * z, dim=-1, keepdim=True)  # [B,1]
        fused_prob = torch.sigmoid(fused_logit)               # [B,1]
        return fused_prob, a

# ===================== 训练与评估 =====================
def train_fuser(merged: pd.DataFrame, device):
    X = merged[["img_prob","txt_prob","time_prob"]].to_numpy()
    y = merged["binary_true"].to_numpy()
    ids = merged["id"].to_numpy()

    # 先切 test
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    train_ds = TrioDataset(X_train, y_train)
    test_ds  = TrioDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 类别不平衡：pos_weight = neg/pos
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    pos_weight = torch.tensor((neg / max(1, pos)) if pos > 0 else 1.0, dtype=torch.float32, device=device)

    model = AttnFusion(hidden_size=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def run_epoch(loader, train_mode=False):
        if train_mode:
            model.train()
        else:
            model.eval()
        losses, probs_all, labels_all, attn_all = [], [], [], []
        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                prob, attn = model(xb)           # prob: [B,1]
                prob = prob.squeeze(1)           # [B]
                # 手工权重 BCE
                loss_pos = - yb * torch.log(torch.clamp(prob, 1e-6, 1-1e-6))
                loss_neg = - (1 - yb) * torch.log(torch.clamp(1 - prob, 1e-6, 1-1e-6))
                loss = (pos_weight * loss_pos + loss_neg).mean()

                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                losses.append(loss.item())
                probs_all.append(prob.detach().cpu().numpy())
                labels_all.append(yb.detach().cpu().numpy())
                attn_all.append(attn.detach().cpu().numpy())

        probs = np.concatenate(probs_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        attn = np.concatenate(attn_all, axis=0)  # [N,3]
        preds = (probs >= THRESH).astype(int)
        acc = accuracy_score(labels, preds)
        p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = float("nan")
        return float(np.mean(losses)), acc, f1_ma, auc, probs, preds, labels, attn

    set_seed(SEED)
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc, tr_f1, tr_auc, *_ = run_epoch(train_loader, train_mode=True)
        print(f"[Epoch {ep:02d}] Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} auc={tr_auc:.4f}")

    te_loss, te_acc, te_f1, te_auc, te_prob, te_pred, te_true, te_attn = run_epoch(test_loader, train_mode=False)

    metrics = {
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_macro_f1": float(te_f1),
        "test_auc": float(te_auc),
        "pos_weight": float(pos_weight.item())
    }
    return model, (X_test, y_test, id_test), (te_prob, te_pred, te_true, te_attn), metrics

# ===================== 可视化 & 保存 =====================
def save_confusion_matrix(y_true, y_pred, path, title="Attention Fusion - Confusion Matrix (Test)"):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(7,6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j,i,str(int(v)),ha="center",va="center",fontsize=12)
    plt.xticks([0,1], ["-1","1(merged)"])
    plt.yticks([0,1], ["-1","1(merged)"])
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

def try_shap_plots(model, X_train, X_test):
    """
    使用 SHAP 对三维输入特征 [img_prob, txt_prob, time_prob] 做解释。
    - 采用 KernelExplainer，背景样本取训练集的 64 条（或更少）。
    - 若未安装 shap，则跳过并打印提示。
    """
    try:
        import shap
    except Exception as e:
        print(f"[WARN] shap 未安装，跳过 SHAP 可视化。建议: pip install shap  (原因: {e})")
        return

    # 包装模型输出 P(y=1|x)
    device = next(model.parameters()).device
    def f_np(x_np):
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            p, _ = model(x)
            return p.squeeze(1).detach().cpu().numpy()

    # 选背景与解释集
    bg = X_train[:min(64, len(X_train))]
    expl = shap.KernelExplainer(f_np, bg)
    sv = expl.shap_values(X_test, nsamples=200)  # 小数据量即可

    # 绘图
    plt.figure()
    shap.summary_plot(sv, X_test, feature_names=["img_prob","txt_prob","time_prob"], show=False)
    plt.tight_layout(); plt.savefig(OUT_SHAP_SUM, dpi=160); plt.close()
    print(f"[INFO] SHAP summary saved -> {OUT_SHAP_SUM}")

    # 条形图（平均 |shap|）
    try:
        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(-mean_abs)
        plt.figure(figsize=(6,4))
        plt.bar([["img_prob","txt_prob","time_prob"][i] for i in order], mean_abs[order])
        plt.title("Mean |SHAP value|"); plt.tight_layout()
        plt.savefig(OUT_SHAP_BAR, dpi=160); plt.close()
        print(f"[INFO] SHAP bar saved -> {OUT_SHAP_BAR}")
    except Exception as e:
        print(f"[WARN] 绘制 SHAP bar 失败: {e}")

def main():
    set_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    merged, used_paths = load_sources()
    print("[INFO] Loaded files:")
    for p in used_paths: print("  -", p)
    print(f"[INFO] Samples after merge: {len(merged)}")
    print(f"[INFO] Using columns: {list(merged.columns)}")

    # 训练 Attention 融合器
    model, test_pack, test_outputs, metrics = train_fuser(merged, DEVICE)
    X_test, y_test, id_test = test_pack
    te_prob, te_pred, te_true, te_attn = test_outputs  # te_attn: [N,3] -> [img, txt, time]

    # 组装 result.csv：取测试子集对应的三路概率与注意力权重
    sub = merged.set_index("id").loc[id_test][["img_prob","txt_prob","time_prob"]].reset_index()
    sub["img_prob_neg"] = 1.0 - sub["img_prob"]
    sub["txt_prob_neg"] = 1.0 - sub["txt_prob"]
    sub["time_prob_neg"] = 1.0 - sub["time_prob"]
    sub["fused_prob"] = te_prob
    sub["fused_prob_neg"] = 1.0 - sub["fused_prob"]
    sub["fused_pred"] = te_pred
    sub["binary_true"] = te_true
    sub["attn_img"]  = te_attn[:,0]
    sub["attn_txt"]  = te_attn[:,1]
    sub["attn_time"] = te_attn[:,2]
    sub.to_csv(OUT_RESULT, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved result table -> {OUT_RESULT}")

    # 混淆矩阵
    save_confusion_matrix(te_true, te_pred, OUT_CM)
    print(f"[INFO] Saved confusion matrix -> {OUT_CM}")

    # 指标 + 报告
    rep = classification_report(te_true, te_pred, target_names=["-1","1(merged)"], digits=4)
    with open(OUT_METRIC, "w", encoding="utf-8") as f:
        f.write("=== Attention-based Fusion Metrics (Test) ===\n\n")
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        f.write("\n\n[Classification Report]\n")
        f.write(rep)
        f.write("\n\n[Notes]\n")
        f.write("- 已对三份 CSV 的 id/概率/真值列做鲁棒识别，常见异常会自动修复（见控制台日志）。\n")
        f.write("- 模型学习每条样本的注意力权重，对三路 logit 做加权融合。\n")
        f.write("- 新增 SHAP 解释（若已安装 shap），用于量化三路特征对融合输出的贡献。\n")

    print(f"[INFO] Saved metrics -> {OUT_METRIC}")

    # ===== SHAP 解释（可选，若未安装 shap 将自动跳过）=====
    # 取训练/测试划分（与 train_fuser 内部一致）的近似方式：从 merged 中重切一次即可用于 SHAP 背景
    X = merged[["img_prob","txt_prob","time_prob"]].to_numpy().astype(np.float32)
    y = merged["binary_true"].to_numpy()
    X_tmp, X_test2, y_tmp, y_test2 = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    if VAL_SIZE and float(VAL_SIZE) > 0:
        X_train2, X_val2, y_train2, y_val2 = train_test_split(
            X_tmp, y_tmp, test_size=VAL_SIZE, random_state=SEED, stratify=y_tmp
        )
    else:
        X_train2 = X_tmp
    try_shap_plots(model, X_train2, X_test2)

    print("[DONE] Attention 融合完成。")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
