# -*- coding: utf-8 -*-
"""
仅用 CSV 第二列（时间）与第七列（分类结果/标签）做拟合，不参与任何图片。
- 输入: ./with_images.filtered.csv
    * 第1列: id
    * 第2列: 时间字符串（格式示例: "01月01日 23:08"）
    * 第7列: 标签（-1/0/1），二值化：-1 -> 0， {0,1} -> 1
- 训练/评估：只使用“时间可解析 & 标签有效”的子集
- **保存结果：不删除任何一行**，保持与原文件相同行数与顺序；第1列为 id
- 输出:
    ./time_fit_no_image/metrics.txt                      指标与报告（基于可训练子集）
    ./time_fit_no_image/coefficients.csv                 模型系数
    ./time_fit_no_image/predictions_with_time_only.csv   全量逐行结果（含 NaN 预测）
    ./time_fit_no_image/effect_hourly.png                按小时的经验正类率与模型趋势（基于可训练子集）
    ./time_fit_no_image/effect_doy.png                   按年内天数的经验正类率与模型趋势（基于可训练子集）
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, classification_report
)
from sklearn.linear_model import LogisticRegression

# ========== 路径 ==========
CSV_PATH = os.environ.get("CSV_PATH", "./with_images.filtered.csv")
OUT_DIR  = os.environ.get("TIME_OUT_DIR", "./time_fit_no_image")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_METRICS = os.path.join(OUT_DIR, "metrics.txt")
OUT_COEF    = os.path.join(OUT_DIR, "coefficients.csv")
OUT_PRED    = os.path.join(OUT_DIR, "predictions_with_time_only.csv")
OUT_HOUR_FG = os.path.join(OUT_DIR, "effect_hourly.png")
OUT_DOY_FG  = os.path.join(OUT_DIR, "effect_doy.png")

# ========== 时间解析 ==========
TIME_RE = re.compile(r"^\s*(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{2})\s*$")

def parse_time_str(s: str):
    """解析 'MM月DD日 HH:MM' -> (month, day, hour, minute)；失败返回 None"""
    if not isinstance(s, str):
        return None
    m = TIME_RE.match(s)
    if not m:
        return None
    mm, dd, HH, MM = map(int, m.groups())
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and 0 <= HH <= 23 and 0 <= MM <= 59):
        return None
    return mm, dd, HH, MM

def day_of_year(month, day, year=2001):
    """给定月/日，返回一年中的第几天（使用平年 2001）；失败返回 None。"""
    try:
        dt = datetime(year=year, month=month, day=day)
        return int(dt.timetuple().tm_yday)
    except Exception:
        return None

def binarize(lbl):
    """-1 -> 0；其余 {0,1} -> 1；其它返回 None"""
    s = str(lbl).strip()
    if s == "-1": return 0
    if s in {"0", "1"}: return 1
    return None

def cyclical(x, period):
    """返回 (sin, cos) 周期特征"""
    ang = 2 * math.pi * (float(x) % period) / period
    return math.sin(ang), math.cos(ang)

# ========== 读取 ==========
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"未找到 CSV 文件: {CSV_PATH}")

df = None
for _enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
    try:
        df = pd.read_csv(CSV_PATH, encoding=_enc)
        break
    except UnicodeDecodeError:
        continue
if df is None:
    raise RuntimeError("CSV编码不支持，请使用 UTF-8 或 GBK 编码保存")

# 取 id, time, label（不删任何行）
if df.shape[1] < 7:
    raise ValueError("CSV 列数不足，需至少包含第1、2、7列。")

orig = df.iloc[:, [0,1,6]].copy()
orig.columns = ["id", "time_str", "raw_label"]

# 计算解析后的时间特征（不删行，保留 NaN）
parsed = orig["time_str"].apply(parse_time_str)
time_ok = parsed.notnull()

# 为所有行创建特征列（失败行填 NaN）
feat = pd.DataFrame(index=orig.index)
if time_ok.any():
    tmp = pd.DataFrame(parsed[time_ok].tolist(), index=orig.index[time_ok], columns=["month","day","hour","minute"])
    feat[["month","day","hour","minute"]] = tmp
else:
    feat[["month","day","hour","minute"]] = np.nan

# day_of_year
feat["doy"] = np.nan
mask_month_day_ok = feat["month"].notnull() & feat["day"].notnull()
feat.loc[mask_month_day_ok, "doy"] = [
    day_of_year(int(m), int(d)) for m, d in zip(feat.loc[mask_month_day_ok, "month"], feat.loc[mask_month_day_ok, "day"])
]

# 周期特征（对 NaN 保持 NaN）
def safe_cyc_col(val, period, which):
    if pd.isna(val): return np.nan
    s, c = cyclical(val, period)
    return s if which == "sin" else c

feat["sin_hour"] = feat["hour"].apply(lambda x: safe_cyc_col(x, 24, "sin"))
feat["cos_hour"] = feat["hour"].apply(lambda x: safe_cyc_col(x, 24, "cos"))
feat["sin_doy"]  = feat["doy"].apply(lambda x: safe_cyc_col(x, 365, "sin"))
feat["cos_doy"]  = feat["doy"].apply(lambda x: safe_cyc_col(x, 365, "cos"))

# 目标标签列（不删除行，可能为 None）
orig["binary_label"] = orig["raw_label"].apply(binarize)

# ========== 训练数据子集（仅用于拟合/评估，不影响最终保存的全量输出）==========
feature_cols = ["hour","minute","month","day","doy","sin_hour","cos_hour","sin_doy","cos_doy"]

train_mask = feat[feature_cols].notnull().all(axis=1) & orig["binary_label"].notnull()
train_idx = np.where(train_mask.values)[0]

if len(train_idx) < 10:
    raise RuntimeError(f"可用于训练的数据太少（仅 {len(train_idx)} 条）。请检查时间格式与标签。")

X_all = feat[feature_cols].astype(float)
y_all = orig["binary_label"].astype("float")

X_train_all = X_all.iloc[train_idx]
y_train_all = y_all.iloc[train_idx].astype(int)

# 标准化（仅基于训练子集）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_all)

TEST_RATIO = float(os.environ.get("TEST_RATIO", os.environ.get("TIME_TEST_RATIO", "0.2")))

# 划分训练/测试（对可训练子集）
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train_scaled, y_train_all.values, test_size=TEST_RATIO, random_state=42, stratify=y_train_all.values
)

# ========== 模型训练 ==========
clf = LogisticRegression(
    C=1.0, solver="liblinear", class_weight="balanced", max_iter=200
)
clf.fit(X_tr, y_tr)

# 评估（在可训练子集的测试划分上）
prob_te = clf.predict_proba(X_te)[:,1]
pred_te = (prob_te >= 0.5).astype(int)
acc = accuracy_score(y_te, pred_te)
p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_te, pred_te, average="macro", zero_division=0)
try:
    auc = roc_auc_score(y_te, prob_te)
except Exception:
    auc = float("nan")
report = classification_report(y_te, pred_te, target_names=["-1","1(merged)"], digits=4)

# 系数导出
coef = clf.coef_.ravel()
coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
coef_df.to_csv(OUT_COEF, index=False, encoding="utf-8-sig")

# ========== 对“全量行”做预测（不删除任何行）==========
# 能够形成完整特征的行→用 scaler.transform，再预测；否则填 NaN
time_feature_ok = feat[feature_cols].notnull().all(axis=1)

time_fit_prob_pos = pd.Series(np.nan, index=orig.index, dtype=float)
time_fit_pred     = pd.Series(np.nan, index=orig.index, dtype=float)

if time_feature_ok.any():
    X_full_ok_scaled = scaler.transform(feat.loc[time_feature_ok, feature_cols].astype(float))
    prob_full_ok = clf.predict_proba(X_full_ok_scaled)[:,1]
    pred_full_ok = (prob_full_ok >= 0.5).astype(int)
    time_fit_prob_pos.loc[time_feature_ok] = prob_full_ok
    time_fit_pred.loc[time_feature_ok]     = pred_full_ok

# 组装“全量逐行输出”：保持原始顺序；第1列为 id；不删除任何行
out = pd.concat([orig[["id","time_str","raw_label","binary_label"]], feat[feature_cols]], axis=1)
out.insert(1, "time_fit_prob_pos", time_fit_prob_pos.values)  # 紧随 id 后，便于对齐
out.insert(2, "time_fit_pred",     time_fit_pred.values)

# 保存
out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")

# 指标保存（基于可训练子集的测试划分；与全量输出无关）
with open(OUT_METRICS, "w", encoding="utf-8") as f:
    f.write("=== 仅用第二列时间与第七列标签的逻辑回归拟合 ===\n\n")
    f.write(f"总样本数(保存时不丢行): {len(df)}\n")
    f.write(f"可训练子集样本数: {len(train_idx)}\n")
    f.write(f"特征: {feature_cols}\n\n")
    f.write("[可训练子集-测试集] 指标：\n")
    f.write(f"  ACC  : {acc:.4f}\n")
    f.write(f"  F1(m): {f1_ma:.4f}\n")
    f.write(f"  AUC  : {auc:.4f}\n\n")
    f.write("[可训练子集-测试集] 分类报告：\n")
    f.write(report + "\n")
    f.write("[模型系数（同 coefficients.csv）]\n")
    for fe, c in zip(feature_cols, coef):
        f.write(f"  {fe:10s}: {c:+.6f}\n")

print(f"[INFO] 指标保存到: {OUT_METRICS}")
print(f"[INFO] 系数保存到: {OUT_COEF}")
print(f"[INFO] 逐样本预测保存到(保留所有行): {OUT_PRED}")

# ========== 可视化：基于“可训练子集” ==========
# 为可解释性，这两张图只用“可训练子集”，不含 NaN 行
subset = pd.concat([orig.loc[train_idx, ["binary_label"]],
                    feat.loc[train_idx, feature_cols]], axis=1)

# 1) 按小时
plt.figure(figsize=(8,6))
hour_grp = subset.groupby("hour")["binary_label"].mean()
plt.plot(hour_grp.index, hour_grp.values, marker="o", label="Empirical positive rate by HOUR (trainable subset)")

hrs = np.arange(24)
med = subset[feature_cols].median()
grid_h = pd.DataFrame({
    "hour": hrs,
    "minute": med["minute"],
    "month": med["month"],
    "day": med["day"],
    "doy": med["doy"],
    "sin_hour": [math.sin(2*math.pi*h/24) for h in hrs],
    "cos_hour": [math.cos(2*math.pi*h/24) for h in hrs],
    "sin_doy": math.sin(2*math.pi*med["doy"]/365),
    "cos_doy": math.cos(2*math.pi*med["doy"]/365),
})
grid_h_scaled = scaler.transform(grid_h[feature_cols])
grid_h_prob = clf.predict_proba(grid_h_scaled)[:,1]
plt.plot(hrs, grid_h_prob, linestyle="--", label="Model trend (vary hour)")
plt.xlabel("Hour of Day"); plt.ylabel("Positive rate / Probability")
plt.title("Time vs Label (Hourly) — trainable subset")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_HOUR_FG, dpi=160); plt.close()
print(f"[INFO] 小时趋势图保存到: {OUT_HOUR_FG}")

# 2) 按 DOY
plt.figure(figsize=(8,6))
doy_grp = subset.groupby("doy")["binary_label"].mean().sort_index()
if len(doy_grp) > 7:
    doy_smooth = doy_grp.rolling(7, min_periods=1, center=True).mean()
else:
    doy_smooth = doy_grp
plt.plot(doy_smooth.index, doy_smooth.values, label="Empirical positive rate by DOY (7d smooth, trainable subset)")

doys = np.arange(int(subset["doy"].min()), int(subset["doy"].max())+1)
grid_d = pd.DataFrame({
    "hour": med["hour"],
    "minute": med["minute"],
    "month": med["month"],
    "day": med["day"],
    "doy": doys,
})
grid_d["sin_hour"] = math.sin(2*math.pi*med["hour"]/24)
grid_d["cos_hour"] = math.cos(2*math.pi*med["hour"]/24)
grid_d["sin_doy"]  = np.sin(2*np.pi*grid_d["doy"]/365.0)
grid_d["cos_doy"]  = np.cos(2*np.pi*grid_d["doy"]/365.0)
grid_d_scaled = scaler.transform(grid_d[feature_cols])
grid_d_prob = clf.predict_proba(grid_d_scaled)[:,1]
plt.plot(doys, grid_d_prob, linestyle="--", label="Model trend (vary DOY)")
plt.xlabel("Day of Year"); plt.ylabel("Positive rate / Probability")
plt.title("Time vs Label (Day-of-Year) — trainable subset")
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(OUT_DOY_FG, dpi=160); plt.close()
print(f"[INFO] 年内天数趋势图保存到: {OUT_DOY_FG}")

print("[DONE] 拟合完成；结果已在不删行的前提下按第一列(id)输出。")
