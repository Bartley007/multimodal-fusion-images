# -*- coding: utf-8 -*-
"""
Chinese BERT 二分类微调（GPU+AMP+tqdm，safetensors-only，Windows 多进程安全）
- 标签合并：{-1}→0（负类），{0,1}→1（正类）
- 数据：CSV 第4列=文本，第7列=标签（仅保留 -1/0/1）
- 训练：类别加权交叉熵、线性 warmup、AMP 混合精度
- 评估：测试集混淆矩阵与报告
- 导出：全量数据“拟合值（正类概率）”CSV
"""

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from contextlib import nullcontext
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from transformers.utils.logging import set_verbosity_info


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


def safe_tokenizer_from_pretrained(model_id: str):
    last_err = None
    # 先离线尝试（本地目录/本地缓存），避免网络不可达时长时间重试
    for src in _candidate_model_sources(model_id):
        try:
            return AutoTokenizer.from_pretrained(src, local_files_only=True)
        except Exception as e:
            last_err = e

    # 再尝试在线
    for src in _candidate_model_sources(model_id):
        try:
            return AutoTokenizer.from_pretrained(src)
        except Exception as e:
            last_err = e

    raise OSError(_hf_offline_hint(model_id) + f"\nOriginal error: {last_err}")


def safe_model_from_pretrained(model_id: str, *, num_labels: int, device: torch.device):
    last_err = None
    # 先离线尝试（本地目录/本地缓存），避免网络不可达时长时间重试
    for src in _candidate_model_sources(model_id):
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                src, num_labels=num_labels, use_safetensors=True, local_files_only=True
            ).to(device)
        except Exception as e:
            last_err = e

    # 再尝试在线
    for src in _candidate_model_sources(model_id):
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                src, num_labels=num_labels, use_safetensors=True
            ).to(device)
        except Exception as e:
            last_err = e

    raise OSError(_hf_offline_hint(model_id) + f"\nOriginal error: {last_err}")

# 设置镜像站（针对中国网络）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[INFO] HF_ENDPOINT not set, using hf-mirror.com")
TEXT_COL_IDX, LABEL_COL_IDX = 3, 6
_ENV_MODEL_ID = os.environ.get("BERT_MODEL_ID")
MODEL_ID = _ENV_MODEL_ID or "AUTO"
OUTPUT_DIR = "./cbert_binary_gpu_out"
BEST_DIR = "./cbert_binary_gpu_best"
RESULT_CSV = "predictions_binary_with_fit.csv"
MAX_LEN = int(os.environ.get("BERT_MAX_LEN", "128"))
BATCH_SIZE = int(os.environ.get("BERT_BATCH_SIZE", "32"))
LR = float(os.environ.get("BERT_LR", "2e-5"))
NUM_EPOCHS = int(os.environ.get("BERT_EPOCHS", "6"))
WEIGHTED_LOSS = True
GRAD_ACC_STEPS = 1
WARMUP_RATIO = 0.06
SEED = 42
NUM_WORKERS = int(os.environ.get("BERT_NUM_WORKERS", "2"))
PREFETCH_FACTOR = int(os.environ.get("BERT_PREFETCH_FACTOR", "2"))
BERT_LABEL_SMOOTH = float(os.environ.get("BERT_LABEL_SMOOTH", "0.0"))
GRAD_CLIP_NORM = float(os.environ.get("BERT_GRAD_CLIP_NORM", "1.0"))
ENABLE_TF32 = os.environ.get("ENABLE_TF32", "1") == "1"
ENABLE_COMPILE = os.environ.get("BERT_USE_COMPILE", "0") == "1"
BERT_AUTO_TUNE = os.environ.get("BERT_AUTO_TUNE", "1") == "1"
# ======================

TEST_RATIO = float(os.environ.get("TEST_RATIO", os.environ.get("BERT_TEST_RATIO", "0.2")))

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_text_language(texts, sample_size: int = 200) -> str:
    """Very lightweight language heuristic.
    Returns: 'zh' for mostly Chinese, 'en' for mostly Latin/English.
    """
    if not texts:
        return "zh"
    n = min(sample_size, len(texts))
    cjk = 0
    latin = 0
    for t in texts[:n]:
        s = str(t)
        cjk += len(_CJK_RE.findall(s))
        latin += len(_LATIN_RE.findall(s))
    if cjk == 0 and latin == 0:
        return "zh"
    return "zh" if cjk >= latin else "en"


def pick_default_model_id(lang: str) -> str:
    # 尝试使用更轻量的多语言模型，或优先使用本地已有的模型
    # 如果本地有 bert-base-chinese，且主要文本是中文，则优先使用
    local_zh = Path(__file__).resolve().parent / "models" / "bert-base-chinese"
    if lang == "zh" and local_zh.is_dir():
        return "bert-base-chinese"
    
    # 默认返回多语言模型以兼顾中英文
    return "bert-base-multilingual-cased"

def set_seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class TextDS(Dataset):
    def __init__(self, X, y=None):
        self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        if self.y is None: return self.X[i], 0
        return self.X[i], int(self.y[i])

class Collator:
    """顶层可序列化的 collate（训练/验证/测试）"""
    def __init__(self, tokenizer_name: str, max_len: int):
        # 注意：在子进程里也会执行 __init__，因此这里直接 by-name 加载 tokenizer
        self.tokenizer = safe_tokenizer_from_pretrained(tokenizer_name)
        self.max_len = max_len
    def __call__(self, batch):
        texts, labels = zip(*batch)
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

class CollatorAll:
    """顶层可序列化的 collate（全量推理，无 label）"""
    def __init__(self, tokenizer_name: str, max_len: int):
        self.tokenizer = safe_tokenizer_from_pretrained(tokenizer_name)
        self.max_len = max_len
    def __call__(self, batch):
        texts, _ = zip(*batch)
        enc = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_len, return_tensors="pt"
        )
        return enc

def binarize(lbl_str: str) -> int:
    return 0 if str(lbl_str) == "-1" else 1

def load_backbone_safetensors_only(model_id: str, num_labels: int, device: torch.device):
    try:
        return safe_model_from_pretrained(model_id, num_labels=num_labels, device=device)
    except Exception as e:
        raise RuntimeError(
            "无法以 safetensors 方式加载模型权重。\n"
            "请更换带 safetensors 的模型仓库（默认 bert-base-chinese），"
            "或改走路线A升级 PyTorch≥2.6 以允许 .bin。\n"
            f"原始错误：{e}"
        )

def evaluate(model, dataloader, device, autocast, desc="Evaluating"):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad(), tqdm(total=len(dataloader), desc=desc, unit="batch") as pbar:
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            with autocast():
                logits = model(**batch).logits
            all_logits.append(logits.float().detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            pbar.update(1)
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]  # 正类(合并1)概率
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1_ma, "weighted_f1": f1_w}, preds, labels, probs, logits


def auto_tune_bert_runtime(use_gpu: bool):
    batch = BATCH_SIZE
    workers = NUM_WORKERS
    prefetch = PREFETCH_FACTOR
    if not BERT_AUTO_TUNE:
        return batch, workers, prefetch
    try:
        if use_gpu and torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
            if total_gb <= 6:
                batch = min(batch, 16)
            elif total_gb <= 8:
                batch = max(batch, 24)
            elif total_gb <= 12:
                batch = max(batch, 32)
            elif total_gb <= 16:
                batch = max(batch, 48)
            else:
                batch = max(batch, 64)
        cpu_n = os.cpu_count() or 4
        workers = min(max(0, cpu_n // 4), 4)
        prefetch = 2 if workers > 0 else PREFETCH_FACTOR
    except Exception:
        pass
    return int(batch), int(workers), int(prefetch)

def main():
    t0_total = time.perf_counter()
    benchmark = {}
    set_verbosity_info()
    set_seed_all(SEED)

    USE_GPU = torch.cuda.is_available()
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"[INFO] Device: {device}")
    if USE_GPU:
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        if ENABLE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    autocast = (lambda: torch.cuda.amp.autocast(enabled=USE_GPU)) if USE_GPU else (lambda: nullcontext())
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_GPU)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_GPU)
    torch.backends.cudnn.benchmark = USE_GPU

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    t0_data = time.perf_counter()
    # 1) 数据
    df = None
    for _enc in [
        "utf-8",
        "utf-8-sig",
        "gbk",
        "gb2312",
        "cp1252",
        "latin1",
        "utf-16",
        "utf-16-le",
        "utf-16-be",
        "big5",
    ]:
        try:
            df = pd.read_csv(CSV_PATH, encoding=_enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise RuntimeError("CSV编码不支持，请使用 UTF-8 或 GBK 编码保存")
    valid = set([-1, 0, 1, "-1", "0", "1"])
    mask = df.iloc[:, LABEL_COL_IDX].isin(valid)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows with invalid labels")
    df = df[mask].reset_index(drop=True)

    texts_all = df.iloc[:, TEXT_COL_IDX].astype(str).tolist()
    raw_labels_all = df.iloc[:, LABEL_COL_IDX].astype(str).tolist()

    global MODEL_ID
    if MODEL_ID == "AUTO":
        lang = detect_text_language(texts_all)
        MODEL_ID = pick_default_model_id(lang)
        print(f"[INFO] Detected text language: {lang} -> model={MODEL_ID}")

    y_all_bin = np.array([binarize(v) for v in raw_labels_all], dtype=int)
    print(f"[INFO] Binary distribution (all): neg(0)={(y_all_bin==0).sum()} | pos(1)={(y_all_bin==1).sum()}")

    # 2) 划分（train/test）
    X_train, X_test, y_train, y_test = train_test_split(
        texts_all, y_all_bin, test_size=TEST_RATIO, random_state=SEED, stratify=y_all_bin
    )
    print(f"[INFO] Split -> train:{len(X_train)} test:{len(X_test)}")
    benchmark["data_prepare_sec"] = round(time.perf_counter() - t0_data, 3)

    tuned_batch_size, tuned_workers, tuned_prefetch = auto_tune_bert_runtime(USE_GPU)
    print(
        f"[INFO] BERT runtime config -> batch={tuned_batch_size}, "
        f"num_workers={tuned_workers}, prefetch={tuned_prefetch}, auto_tune={BERT_AUTO_TUNE}"
    )

    # 3) DataLoader（使用顶层 Collator，子进程可pickle）
    print("[STEP] creating dataloaders...", flush=True)
    collate = Collator(MODEL_ID, MAX_LEN)
    collate_all = CollatorAll(MODEL_ID, MAX_LEN)
    dl_common = {
        "pin_memory": USE_GPU,
        "num_workers": tuned_workers,
    }
    if tuned_workers > 0:
        dl_common["persistent_workers"] = True
        dl_common["prefetch_factor"] = tuned_prefetch

    train_loader = DataLoader(TextDS(X_train, y_train), batch_size=tuned_batch_size, shuffle=True,
                              collate_fn=collate, **dl_common)
    test_loader  = DataLoader(TextDS(X_test,  y_test),  batch_size=tuned_batch_size, shuffle=False,
                              collate_fn=collate, **dl_common)

    # 4) 类别权重
    class_weights = None
    if WEIGHTED_LOSS:
        counts = np.bincount(y_train, minlength=2).astype(float)
        inv = 1.0 / np.maximum(counts, 1.0)
        class_weights = torch.tensor(inv / inv.sum() * 2, dtype=torch.float32, device=device)
        print(f"[INFO] Class counts(train): neg={counts[0]} pos={counts[1]}")
        print(f"[INFO] Class weights: {class_weights.tolist()}")

    # 5) 模型（safetensors-only）
    t0_model = time.perf_counter()
    print("[STEP] loading model (safetensors only)...", flush=True)
    model = load_backbone_safetensors_only(MODEL_ID, num_labels=2, device=device)
    if ENABLE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled for BERT model")
        except Exception as e:
            print(f"[WARN] torch.compile unavailable, fallback to eager mode: {e}")
    benchmark["model_prepare_sec"] = round(time.perf_counter() - t0_model, 3)

    # 优化器、调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optim_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LR)

    est_steps_per_epoch = max(1, len(train_loader) // GRAD_ACC_STEPS)
    num_training_steps = max(1, NUM_EPOCHS * est_steps_per_epoch)
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    criterion = (
        nn.CrossEntropyLoss(weight=class_weights, label_smoothing=BERT_LABEL_SMOOTH)
        if class_weights is not None
        else nn.CrossEntropyLoss(label_smoothing=BERT_LABEL_SMOOTH)
    )

    # 6) 训练
    t0_train = time.perf_counter()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{NUM_EPOCHS} - Training", unit="batch") as pbar:
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                labels = batch.pop("labels")

                with autocast():
                    outputs = model(**batch)
                    loss = criterion(outputs.logits, labels)
                    loss = loss / GRAD_ACC_STEPS

                scaler.scale(loss).backward()

                if step % GRAD_ACC_STEPS == 0:
                    if GRAD_CLIP_NORM > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * GRAD_ACC_STEPS
                pbar.set_postfix({"loss": f"{running_loss/step:.4f}"})
                pbar.update(1)

        # 每轮简单打印 train loss，不再使用验证集选 best
        print(f"[Epoch {epoch}] Train done")
    benchmark["train_sec"] = round(time.perf_counter() - t0_train, 3)

    # 保存 final 模型
    print(f"[INFO] Training finished. Saving final model to {BEST_DIR}")
    model.save_pretrained(BEST_DIR)
    safe_tokenizer_from_pretrained(MODEL_ID).save_pretrained(BEST_DIR)
    model.config.id2label = {0: "-1", 1: "(0|1)"}
    model.config.label2id = {"-1": 0, "(0|1)": 1}
    model.config.save_pretrained(BEST_DIR)

    # 7) 测试集评估 + 混淆矩阵
    t0_eval = time.perf_counter()
    best_model = AutoModelForSequenceClassification.from_pretrained(
        BEST_DIR, use_safetensors=True
    ).to(device)
    model = best_model

    test_metrics, test_preds, test_labels, test_probs, _ = evaluate(model, test_loader, device, autocast, desc="Testing")
    print("\n[TEST] metrics:", test_metrics)
    print("\n[TEST] Classification report (binary, -1 vs {0|1}):")
    print(classification_report(test_labels, test_preds, target_names=["-1", "1(merged)"], digits=4))

    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Chinese BERT Binary (GPU, safetensors) - Confusion Matrix (Test)\n(-1 vs {0|1})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center", fontsize=11)
    plt.xticks([0, 1], ["-1", "1(merged)"]); plt.yticks([0, 1], ["-1", "1(merged)"])
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "cm_test_binary_gpu.png"), dpi=160); plt.show()
    print(f"[INFO] Confusion matrix saved to {os.path.join(OUTPUT_DIR, 'cm_test_binary_gpu.png')}")
    benchmark["eval_sec"] = round(time.perf_counter() - t0_eval, 3)

    # 8) 全量推理并导出拟合值
    t0_infer = time.perf_counter()
    all_loader = DataLoader(TextDS(texts_all, None), batch_size=tuned_batch_size, shuffle=False,
                            collate_fn=collate_all, **dl_common)

    model.eval()
    all_probs_pos = []
    all_probs_neg = []
    with torch.no_grad(), tqdm(total=len(all_loader), desc="Inferring on ALL", unit="batch") as pbar:
        for enc in all_loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with autocast():
                logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            all_probs_neg.extend(probs[:, 0].float().detach().cpu().numpy().tolist())
            all_probs_pos.extend(probs[:, 1].float().detach().cpu().numpy().tolist())
            pbar.update(1)

    all_probs_pos = np.array(all_probs_pos, dtype=float)
    all_probs_neg = np.array(all_probs_neg, dtype=float)
    all_preds = (all_probs_pos >= 0.5).astype(int)

    out_df = df.copy()
    out_df["binary_true"] = y_all_bin
    out_df["fit_prob_pos"] = all_probs_pos
    out_df["fit_prob_neg"] = all_probs_neg
    out_df["binary_pred"] = all_preds

    out_path = os.path.join(OUTPUT_DIR, RESULT_CSV)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved: {out_path}")
    benchmark["infer_all_sec"] = round(time.perf_counter() - t0_infer, 3)
    benchmark["total_sec"] = round(time.perf_counter() - t0_total, 3)
    benchmark["batch_size"] = tuned_batch_size
    benchmark["num_workers"] = tuned_workers
    benchmark["prefetch_factor"] = tuned_prefetch
    benchmark["test_macro_f1"] = float(test_metrics["macro_f1"])
    benchmark["test_accuracy"] = float(test_metrics["accuracy"])
    benchmark_path = os.path.join(OUTPUT_DIR, "benchmark_bert.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Benchmark saved: {benchmark_path}")

# ---------------- Windows 多进程安全入口 ----------------
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    # mp.set_start_method("spawn", force=True)  # 可保留注释；需要时打开
    main()
