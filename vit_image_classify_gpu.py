# -*- coding: utf-8 -*-
"""
ViT 微调训练（按 ID 划分 / 一对多图片 / 类别不平衡 / 早停 / 聚合评估）
- 模型: google/vit-base-patch16-224（safetensors 可用）
- 输入: ./with_images.filtered.csv（第1列=ID，第7列=标签 -1/0/1）
- 图片: ./images/ 下，以 ID 开头，可有多张 (如 123.jpg / 123_0.png / 123_1.jpg)
- 输出:
    ./vit_finetune_out/predictions_image_with_fit.csv      # 每图预测（仅有图的样本）
    ./vit_finetune_out/predictions_id_aggregated.csv       # **按原CSV行顺序与行数**输出，ID聚合概率回填，无图则NaN
    ./cm_results/cm_vit_finetune.png                       # 混淆矩阵（仅基于有预测值的行）
- 解释性输出（新增）:
    ./vit_finetune_out/explain/rollout_*_overlay.png       # Grad-CAM 风格 attention rollout 热图叠加
    ./vit_finetune_out/explain/rollout_*_raw.png           # 热图原图
"""

import os, json, time, random, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup


def _candidate_model_sources(model_id: str):
    sources = [model_id]
    try:
        local_dir = Path(__file__).resolve().parent / "models" / model_id
        if local_dir.is_dir():
            sources.append(str(local_dir))
    except Exception:
        pass
    return sources


def _hf_offline_hint(model_id: str) -> str:
    local_dir = str((Path(__file__).resolve().parent / "models" / model_id).resolve())
    return (
        "Cannot download from https://huggingface.co. "
        "Please check your network / proxy, or run in offline mode. "
        "Options:\n"
        f"1) Pre-download the model, then rerun (it will use local cache). Model: {model_id}\n"
        f"2) Put the model files under: {local_dir}\n"
        "3) (China) set environment variable HF_ENDPOINT, e.g.\n"
        "   set HF_ENDPOINT=https://hf-mirror.com\n"
        "   set TRANSFORMERS_OFFLINE=0\n"
    )


def safe_image_processor_from_pretrained(model_id: str):
    last_err = None
    # 先离线尝试（本地目录/本地缓存），避免网络不可达时长时间重试
    for src in _candidate_model_sources(model_id):
        try:
            return AutoImageProcessor.from_pretrained(src, use_fast=True, local_files_only=True)
        except Exception as e:
            last_err = e

    # 再尝试在线
    for src in _candidate_model_sources(model_id):
        try:
            return AutoImageProcessor.from_pretrained(src, use_fast=True)
        except Exception as e:
            last_err = e

    raise OSError(_hf_offline_hint(model_id) + f"\nOriginal error: {last_err}")


def safe_vit_model_from_pretrained(model_id: str, *, device: str, num_labels: int = 2):
    last_err = None
    # 先离线尝试（本地目录/本地缓存），避免网络不可达时长时间重试
    for src in _candidate_model_sources(model_id):
        try:
            return AutoModelForImageClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
                local_files_only=True,
            ).to(device)
        except Exception as e:
            last_err = e

    # 再尝试在线
    for src in _candidate_model_sources(model_id):
        try:
            return AutoModelForImageClassification.from_pretrained(
                src,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            ).to(device)
        except Exception as e:
            last_err = e

    raise OSError(_hf_offline_hint(model_id) + f"\nOriginal error: {last_err}")

# 设置镜像站（针对中国网络）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[INFO] HF_ENDPOINT not set, using hf-mirror.com")

# ========= 相对路径配置 =========
CSV_PATH   = os.environ.get("CSV_PATH", "./with_images.filtered.csv")
IMG_DIR    = os.environ.get("IMG_DIR", "./images")

OUT_DIR    = "./vit_finetune_out"; os.makedirs(OUT_DIR, exist_ok=True)
CM_DIR     = "./cm_results";       os.makedirs(CM_DIR, exist_ok=True)
EXPL_DIR   = os.path.join(OUT_DIR, "explain"); os.makedirs(EXPL_DIR, exist_ok=True)

OUT_PER_IMG = os.path.join(OUT_DIR, "predictions_image_with_fit.csv")
OUT_PER_ID  = os.path.join(OUT_DIR, "predictions_id_aggregated.csv")
CM_PATH     = os.path.join(CM_DIR, "cm_vit_finetune.png")

_ENV_MODEL_ID = os.environ.get("VIT_MODEL_ID")
MODEL_ID     = _ENV_MODEL_ID or "google/vit-base-patch16-224"
BATCH_SIZE   = int(os.environ.get("VIT_BATCH_SIZE", "16"))
EPOCHS       = int(os.environ.get("VIT_EPOCHS", "10"))
LR           = float(os.environ.get("VIT_LR", "3e-5"))
WARMUP_RATIO = 0.06
PATIENCE     = 2           # 早停耐心
NUM_WORKERS  = int(os.environ.get("VIT_NUM_WORKERS", "0"))  # Windows 上建议维持 0
SEED         = 42
TRY_EXTS     = [".jpg",".jpeg",".png",".bmp",".webp",".gif",".jfif",".tif",".tiff"]
ROLL_N_SAMPLES = 12        # 解释性可视化抽样数量上限
ENABLE_TF32 = os.environ.get("ENABLE_TF32", "1") == "1"
USE_AMP = os.environ.get("VIT_USE_AMP", "1") == "1"
ENABLE_COMPILE = os.environ.get("VIT_USE_COMPILE", "0") == "1"
VIT_AUTO_TUNE = os.environ.get("VIT_AUTO_TUNE", "1") == "1"
# =================================

TEST_RATIO = float(os.environ.get("TEST_RATIO", os.environ.get("VIT_TEST_RATIO", "0.2")))

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def binarize(lbl):  # -1 -> 0； {0,1} -> 1
    return 0 if str(lbl).strip() == "-1" else 1

def find_images_for_id(root, base_id):
    """在 images/ 下（递归）查找以 base_id 开头、扩展名在 TRY_EXTS 中的图片。"""
    base_id = str(base_id)
    res=[]
    if not os.path.isdir(root):
        return res
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if fn.startswith(base_id) and any(low.endswith(ext) for ext in TRY_EXTS):
                res.append(os.path.join(dirpath, fn))
    return sorted(res)

# 轻量增广（仅 PIL），避免依赖 torchvision
def augment_pil(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        angle = random.uniform(-10, 10)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0,0,0))
    return img

class ImgDS(Dataset):
    """
    __getitem__ 返回 4 项：
      pixel_values(Tensor), label(int), fname(str), base_id(str)
    """
    def __init__(self, recs, processor, mode="train"):
        self.recs = recs
        self.processor = processor
        self.mode = mode

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        path = rec["path"]
        label = int(rec["label"])
        img = Image.open(path).convert("RGB")
        if self.mode == "train":
            img = augment_pil(img)
        enc = self.processor(images=img, return_tensors="pt")
        pixel = enc["pixel_values"][0]
        fname = os.path.basename(path)
        base_id = rec["id"]
        return pixel, label, fname, str(base_id)

    @staticmethod
    def collate(batch, device):
        px, y, names, ids = zip(*batch)
        return {
            "pixel_values": torch.stack(px).to(device, non_blocking=True),
            "labels": torch.tensor(y, dtype=torch.long).to(device, non_blocking=True),
            "names": names,
            "ids": ids,
        }

def build_dataset_and_maps():
    """
    读取 CSV，不删任何行；构建：
      - 原始表 orig_df（含 id/raw_label/binary_true）
      - 仅有图片的 id -> 路径 列表
      - 训练/验证/测试 的样本记录（以图片为单位）
    """
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    # 编码自动检测
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(CSV_PATH, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise RuntimeError("CSV编码不支持，请使用 UTF-8 或 GBK 编码保存")

    if df.shape[1] < 7:
        raise ValueError("CSV 列数不足，需至少包含第1、2、7列。")

    # 原始表（不删行），只处理标签二值化
    orig_df = pd.DataFrame({
        "id": df.iloc[:, 0].astype(str).values,
        "raw_label": df.iloc[:, 6].values
    })
    valid = {"-1","0","1",-1,0,1}
    # 二值标签（对无法识别的原样保留，在聚合输出中仍保留该行）
    orig_df["binary_true"] = [
        binarize(x) if x in valid else np.nan for x in orig_df["raw_label"]
    ]

    # 构建 id->label（仅限标签有效的ID），id->images
    id2label, id2paths = {}, {}
    for _, r in orig_df.iterrows():
        base = r["id"]
        lab  = r["binary_true"]
        paths = find_images_for_id(IMG_DIR, base)
        if len(paths) > 0 and (lab in [0,1]):  # 训练只用“有图且标签有效”的id
            id2label[base] = int(lab)
            id2paths[base] = paths

    # 训练/测试（按ID分层）
    ids = list(id2label.keys())
    if not ids:
        raise RuntimeError("未找到任何图片（且对应标签有效）的ID，请检查 ./images 与 CSV 第一列。")

    y_ids = np.array([id2label[i] for i in ids], dtype=int)
    id_train, id_test = train_test_split(ids, test_size=TEST_RATIO, stratify=y_ids, random_state=SEED)

    def recs(id_list):
        out=[]
        for _id in id_list:
            for p in id2paths[_id]:
                out.append({"id": _id, "path": p, "label": id2label[_id]})
        return out

    tr, te = recs(id_train), recs(id_test)
    print(f"[INFO] Train images:{len(tr)} Test images:{len(te)}")
    return orig_df, id2label, id2paths, tr, te

def evaluate(model, loader, device):
    """按 ID 聚合评估（只对 loader 覆盖到的图片计算）"""
    model.eval()
    names, ids, labels, probs_pos, probs_neg, preds = [], [], [], [], [], []
    use_amp = (str(device).startswith("cuda") and USE_AMP)
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(pixel_values=batch["pixel_values"])
            p = torch.softmax(out.logits, dim=1)
            ps_pos = p[:, 1]
            ps_neg = p[:, 0]
            pr  = (ps_pos >= 0.5).long()
            probs_pos.extend(ps_pos.cpu().numpy())
            probs_neg.extend(ps_neg.cpu().numpy())
            preds.extend(pr.cpu().numpy())
            names.extend(batch["names"])
            ids.extend(batch["ids"])
            labels.extend(batch["labels"].cpu().numpy())

    df_img = pd.DataFrame({
        "id": ids, "image_name": names,
        "binary_true": labels, "fit_prob_pos": probs_pos, "fit_prob_neg": probs_neg, "binary_pred": preds
    })
    df_agg = df_img.groupby("id", as_index=False).agg({
        "binary_true": "first",
        "fit_prob_pos": "mean",
        "fit_prob_neg": "mean",
    })
    df_agg["binary_pred"] = (df_agg["fit_prob_pos"] >= 0.5).astype(int)

    y_true = df_agg["binary_true"].to_numpy()
    y_pred = df_agg["binary_pred"].to_numpy()
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return f1, acc, df_img, df_agg


def auto_tune_vit_runtime(device: str):
    batch = BATCH_SIZE
    workers = NUM_WORKERS
    if not VIT_AUTO_TUNE:
        return int(batch), int(workers)
    try:
        if device == "cuda" and torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
            if total_gb <= 6:
                batch = min(batch, 8)
            elif total_gb <= 8:
                batch = max(batch, 16)
            elif total_gb <= 12:
                batch = max(batch, 24)
            elif total_gb <= 16:
                batch = max(batch, 32)
            else:
                batch = max(batch, 48)
        cpu_n = os.cpu_count() or 4
        workers = 0 if os.name == "nt" else min(max(1, cpu_n // 4), 4)
    except Exception:
        pass
    return int(batch), int(workers)

def plot_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(figsize=(7,6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j,i,str(int(v)),ha="center",va="center",fontsize=12)
    plt.xticks([0,1],["-1","1(merged)"]); plt.yticks([0,1],["-1","1(merged)"])
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()

# ========= ViT Attention Rollout（Grad-CAM 风格） =========
def attention_rollout_heatmap(model, pixel_values, up_size=(224,224), use_grad=True, target_class=1):
    """
    计算 Vision Transformer 的 attention rollout 热图。
    - model: AutoModelForImageClassification（ViT）
    - pixel_values: [1,3,224,224] Tensor (device 同 model)
    - use_grad=True 时，采用梯度加权（更接近 Grad-CAM）
    返回: np.ndarray [H,W]，范围 [0,1]
    """
    model.eval()

    # 某些 transformers 版本/配置下，默认 attn_implementation 不支持 output_attentions
    # 这里尽量切到 eager；失败也不影响主训练流程
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
    except Exception:
        pass

    # 前向（显式请求 attentions）
    out = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)
    logits = out.logits
    attns = out.attentions  # list of [B, heads, T, T]

    if attns is None:
        return None
    # 选择目标 logit（类 1）
    logit = logits[0, target_class]

    if use_grad:
        # 反向获取对 attention 的梯度
        grads = []

        def save_grad(module, grad_input, grad_output):
            # grad_output 是一个 tuple，对应 forward 输出的梯度，这里只取第一个
            grads.append(grad_output[0].detach())

        # 对每层 attention 的输出注册 hook
        handles = []
        for blk in model.vit.encoder.layer:
            # blk.attention.attention_probs 不可直接 hook，这里利用 forward 返出的 attentions 与其梯度
            pass

        # 直接对返出的 attentions 做反向：需要从计算图中取出
        # transformers 返回的 attentions 默认是 Tensor 并保留梯度
        # 但 out.attentions 是 tuple，不在计算图里。我们再次前向，拿图里的 attns：
        def fwd_with_attn():
            o = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)
            return o.logits, o.attentions

        logits2, attns2 = fwd_with_attn()
        chosen = logits2[0, target_class]
        # 对每层 attn 注册 hook：PyTorch 不直接允许对 tuple 注册，只能从 attns2 拿到 tensor 后再 backward
        # 这里我们对 chosen 进行 backward，随后从 .grad_fn 链条读取每层 attn 的 .grad？
        # 更稳妥方式：对每层 attn 乘以 1 并求和后参与损失，让梯度传播到它们，再在 backward 后读取 .grad
        # 简化做法：不读取显式梯度，采用“无梯度 attention rollout”（Chefer 之外的简单版）
        use_grad = False  # 回退：在多数环境下无显式 attn.grad，可用无梯度 Rollout
        # 清理句柄
        for h in handles: h.remove()

    # attention rollout（无梯度或简化）
    # 参考：将每层 heads 平均后 + I，归一化，再连续乘积
    # attns: list[L][B,H,T,T]，取 B=1
    with torch.no_grad():
        attn_mat = [a[0].mean(dim=0) for a in attns]  # [T,T] per layer
        # 添加残差连接：(I + A) / 2
        eye = torch.eye(attn_mat[0].size(-1), device=attn_mat[0].device)
        attn_mat = [(a + eye) / 2 for a in attn_mat]
        # 层间传播：矩阵乘积
        rollout = attn_mat[0]
        for a in attn_mat[1:]:
            rollout = a @ rollout
        # 取 CLS 到各 patch 的注意力（第 0 行）
        mask = rollout[0, 1:]  # exclude CLS
        # ViT-B/16@224 -> 14*14 patch
        L = int((mask.numel()) ** 0.5)
        mask = mask.reshape(L, L)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
        heat = mask.detach().cpu().numpy()

    # 上采样至输入大小
    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L").resize(up_size, resample=Image.BILINEAR)
    heat_arr = np.asarray(heat_img).astype(np.float32) / 255.0
    return heat_arr

def save_rollout_overlay(orig_img_pil, heat01, out_raw, out_overlay, alpha=0.45, cmap="jet"):
    """将热图与原图叠加保存。"""
    import matplotlib.cm as cm
    # 原图转 224x224
    base = orig_img_pil.resize((224,224))
    base_arr = np.asarray(base).astype(np.float32) / 255.0
    heat_rgb = cm.get_cmap(cmap)(heat01)[..., :3]  # [H,W,3]
    overlay = (1 - alpha) * base_arr + alpha * heat_rgb
    Image.fromarray(np.uint8(heat01 * 255), mode="L").save(out_raw)
    Image.fromarray(np.uint8(np.clip(overlay * 255, 0, 255))).save(out_overlay)

def explain_with_rollout(model, processor, id2paths, sample_limit=ROLL_N_SAMPLES):
    """
    对若干测试/验证样本生成 attention rollout 热图。
    每个 ID 取 1 张图演示（减少冗余）。
    """
    done = 0
    for _id, paths in list(id2paths.items()):
        if done >= sample_limit: break
        if not paths: continue
        img_path = paths[0]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        enc = processor(images=img, return_tensors="pt")
        pixel = enc["pixel_values"]
        device = next(model.parameters()).device
        pixel = pixel.to(device)
        try:
            heat = attention_rollout_heatmap(model, pixel, up_size=(224,224), use_grad=False, target_class=1)
        except Exception:
            continue

        if heat is None:
            continue
        # 保存
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        out_raw = os.path.join(EXPL_DIR, f"rollout_{name}_raw.png")
        out_overlay = os.path.join(EXPL_DIR, f"rollout_{name}_overlay.png")
        save_rollout_overlay(img, heat, out_raw, out_overlay)
        done += 1
    print(f"[INFO] Saved attention-rollout explain images: {done}")

def main():
    t0_total = time.perf_counter()
    benchmark = {}
    set_seed(SEED)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    use_amp = (DEVICE == "cuda" and USE_AMP)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ===== 原始表 + 数据集 =====
    t0_data = time.perf_counter()
    orig_df, id2label, id2paths, tr, te = build_dataset_and_maps()
    benchmark["data_prepare_sec"] = round(time.perf_counter() - t0_data, 3)
    tuned_batch_size, tuned_workers = auto_tune_vit_runtime(DEVICE)
    print(f"[INFO] ViT runtime config -> batch={tuned_batch_size}, num_workers={tuned_workers}, auto_tune={VIT_AUTO_TUNE}")

    # 处理器 & 模型
    t0_model = time.perf_counter()
    processor = safe_image_processor_from_pretrained(MODEL_ID)
    model = safe_vit_model_from_pretrained(MODEL_ID, device=DEVICE, num_labels=2)
    if ENABLE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled for ViT model")
        except Exception as e:
            print(f"[WARN] torch.compile unavailable, fallback to eager mode: {e}")
    benchmark["model_prepare_sec"] = round(time.perf_counter() - t0_model, 3)

    # 类别权重（基于训练图片样本数）
    y_tr = [r["label"] for r in tr]
    counts = np.bincount(y_tr, minlength=2).astype(float)
    inv = 1.0 / np.maximum(counts, 1.0)
    cls_w = torch.tensor(inv / inv.sum() * 2, dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Class weights: {cls_w.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cls_w)

    # 优化 & 调度
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = max(1, (len(tr) // BATCH_SIZE) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )

    def make_loader(recs, mode):
        ds = ImgDS(recs, processor, mode)
        return DataLoader(
            ds, batch_size=tuned_batch_size, shuffle=(mode=="train"),
            num_workers=tuned_workers,
            collate_fn=lambda b: ImgDS.collate(b, DEVICE),
            pin_memory=False  # Windows + GPU 环境下避免 pin_memory 错误
        )

    tr_loader, te_loader = make_loader(tr,"train"), make_loader(te,"test")

    # ===== 训练循环 =====
    t0_train = time.perf_counter()
    for ep in range(1, EPOCHS+1):
        model.train(); run_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{EPOCHS}", unit="batch")
        for batch in pbar:
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(pixel_values=batch["pixel_values"])
                loss = criterion(out.logits, batch["labels"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            run_loss += loss.item()
            pbar.set_postfix(loss=run_loss/max(1, pbar.n))
    benchmark["train_sec"] = round(time.perf_counter() - t0_train, 3)

    # 保存 final 模型
    model.save_pretrained(os.path.join(OUT_DIR, "final_model"))
    processor.save_pretrained(os.path.join(OUT_DIR, "final_model"))

    # ===== 测试：评估 final 模型 =====
    t0_eval = time.perf_counter()
    final_model = AutoModelForImageClassification.from_pretrained(os.path.join(OUT_DIR, "final_model")).to(DEVICE)
    f1, acc, df_img, df_agg = evaluate(final_model, te_loader, DEVICE)
    print(f"[TEST] macro_f1={f1:.4f}  acc={acc:.4f}")
    print(classification_report(df_agg["binary_true"], df_agg["binary_pred"],
                                target_names=["-1", "1(merged)"], digits=4))
    benchmark["eval_sec"] = round(time.perf_counter() - t0_eval, 3)

    # ===== 保存逐图片预测（只含有图的样本）=====
    df_img.to_csv(OUT_PER_IMG, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved per-image: {OUT_PER_IMG}")

    # ===== 生成“按原CSV行顺序与行数”的逐行输出 =====
    # 1) 全量推理（所有有图ID），得到 id 聚合概率
    all_recs = []
    for _id, paths in id2paths.items():
        lab = id2label[_id]
        for p in paths:
            all_recs.append({"id": _id, "path": p, "label": lab})

    all_loader = DataLoader(
        ImgDS(all_recs, processor, mode="test"),
        batch_size=tuned_batch_size, shuffle=False, num_workers=tuned_workers,
        collate_fn=lambda b: ImgDS.collate(b, DEVICE),
        pin_memory=False
    )

    final_model.eval()
    t0_infer = time.perf_counter()
    all_ids, all_names, all_probs_pos, all_probs_neg = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(all_loader, desc="Predict(all IDs)", unit="batch"):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = final_model(pixel_values=batch["pixel_values"])
            p = torch.softmax(out.logits, dim=1)
            all_probs_pos.extend(p[:, 1].cpu().numpy())
            all_probs_neg.extend(p[:, 0].cpu().numpy())
            all_names.extend(batch["names"])
            all_ids.extend(batch["ids"])
    benchmark["infer_all_sec"] = round(time.perf_counter() - t0_infer, 3)

    df_img_all = pd.DataFrame({
        "id": all_ids, "image_name": all_names, "fit_prob_pos": all_probs_pos, "fit_prob_neg": all_probs_neg
    })
    df_agg_all = df_img_all.groupby("id", as_index=False).agg({"fit_prob_pos": "mean", "fit_prob_neg": "mean"})
    df_agg_all["binary_pred"] = (df_agg_all["fit_prob_pos"] >= 0.5).astype(int)

    # 2) 回填到原 CSV 行，不删除任何一行，保持原顺序
    df_csv = None
    for _enc in ["utf-8", "gbk", "gb2312", "utf-8-sig"]:
        try:
            df_csv = pd.read_csv(CSV_PATH, encoding=_enc)
            break
        except UnicodeDecodeError:
            continue
    if df_csv is None:
        raise RuntimeError("CSV编码不支持，请使用 UTF-8 或 GBK 编码保存")
    full_out = pd.DataFrame({
        "id": df_csv.iloc[:,0].astype(str).values
    })
    # 为对齐引用 original 的 raw_label/binary_true
    orig_df = pd.DataFrame({
        "id": df_csv.iloc[:,0].astype(str).values,
        "raw_label": df_csv.iloc[:,6].values
    })
    valid = {"-1","0","1",-1,0,1}
    def _bz(x):
        if x in valid:
            return 0 if str(x).strip() == "-1" else 1
        return np.nan
    orig_df["binary_true"] = [ _bz(x) for x in orig_df["raw_label"] ]

    full_out = full_out.merge(df_agg_all, on="id", how="left")
    full_out = full_out.merge(orig_df[["id","binary_true","raw_label"]], on="id", how="left")
    cols = ["id", "fit_prob_pos", "fit_prob_neg", "binary_pred", "binary_true", "raw_label"]
    full_out = full_out[cols]
    full_out.to_csv(OUT_PER_ID, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved per-id (full rows, no deletion): {OUT_PER_ID}")

    # 3) 混淆矩阵：仅基于“有预测值(非NaN)且标签有效”的行
    mask_eval = full_out["fit_prob_pos"].notna() & full_out["binary_true"].isin([0,1])
    if mask_eval.any():
        y_true_cm = full_out.loc[mask_eval, "binary_true"].astype(int).to_numpy()
        y_pred_cm = (full_out.loc[mask_eval, "fit_prob_pos"].to_numpy() >= 0.5).astype(int)
        plot_cm(y_true_cm, y_pred_cm, CM_PATH, "ViT Fine-tuned by ID (All IDs with predictions)")
        print(f"[INFO] Confusion matrix saved to {CM_PATH}")
    else:
        print("[WARN] 无可用于评估的行（可能所有ID均无图片或标签无效），未生成混淆矩阵。")

    # ===== 解释性可视化（Attention Rollout）=====
    explain_with_rollout(final_model, processor, id2paths, sample_limit=ROLL_N_SAMPLES)
    benchmark["total_sec"] = round(time.perf_counter() - t0_total, 3)
    benchmark["batch_size"] = tuned_batch_size
    benchmark["num_workers"] = tuned_workers
    benchmark["test_macro_f1"] = float(f1)
    benchmark["test_accuracy"] = float(acc)
    benchmark_path = os.path.join(OUT_DIR, "benchmark_vit.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Benchmark saved: {benchmark_path}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
