import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import zipfile
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS


APP_ROOT = Path(__file__).resolve().parent
UPLOADS_DIR = APP_ROOT / "uploads"
RESULTS_DIR = APP_ROOT / "results"
FUSION_RESULT_PATH = APP_ROOT / "3 in 1" / "fusion_out" / "result.csv"

VIT_SCRIPT = APP_ROOT / "vit_image_classify_gpu.py"
BERT_SCRIPT = APP_ROOT / "cbert_words.py"
TIME_SCRIPT = APP_ROOT / "time.py"
FUSION_SCRIPT = APP_ROOT / "3 in 1" / "attention_fusion_fit.py"


ALLOWED_CSV_EXTS = {".csv"}
ALLOWED_ARCHIVE_EXTS = {".zip"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".jfif", ".tif", ".tiff"}


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, Dict] = {}


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_filename(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^0-9a-zA-Z._-]+", "_", name)
    return name or "file"


def _json_error(message: str, http_status: int = 400, details: Optional[Dict] = None):
    payload = {"status": "error", "error": message}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), http_status


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _append_job_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.setdefault("logs", [])
        logs: List[str] = job["logs"]
        logs.append(line)
        if len(logs) > 2000:
            job["logs"] = logs[-2000:]


def _set_job(job_id: str, **kwargs) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)


def _run_script(job_id: str, script_path: Path, cwd: Path, env: Dict[str, str], step_name: str) -> None:
    cmd = [sys.executable, str(script_path)]
    _append_job_log(job_id, f"[RUN] {step_name}: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _append_job_log(job_id, line.rstrip("\n"))
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"{step_name} failed(={code})")


def _work_dir_for_upload(upload_id: str) -> Path:
    return UPLOADS_DIR / str(upload_id) / "work"


def _prepare_workdir(upload_id: str) -> Tuple[Path, Path, Path]:
    upload_dir = UPLOADS_DIR / str(upload_id)
    if not upload_dir.is_dir():
        raise FileNotFoundError("upload_iddoes not exist or was cleaned")

    images_dir = upload_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError("not found images directory")

    csv_files = [p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        raise FileNotFoundError("not founduploadCSVfile")
    csv_path = csv_files[0]

    work_dir = _work_dir_for_upload(upload_id)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    work_images_dir = work_dir / "images"
    shutil.copytree(images_dir, work_images_dir)

    work_csv_path = work_dir / "with_images.filtered.csv"
    shutil.copy2(csv_path, work_csv_path)

    return work_dir, work_images_dir, work_csv_path


def _train_pipeline(job_id: str, upload_id: str, config: Dict) -> None:
    try:
        _set_job(job_id, status="running", progress=1, step="prepare")
        work_dir, _, work_csv_path = _prepare_workdir(upload_id)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        vit_epochs = str(int(config.get("vit_epochs", 3)))
        vit_batch = str(int(config.get("vit_batch_size", 32)))
        vit_lr = str(float(config.get("vit_lr", 3e-5)))
        vit_auto_tune = "1" if bool(config.get("vit_auto_tune", False)) else "0"

        bert_epochs = str(int(config.get("bert_epochs", 2)))
        bert_batch = str(int(config.get("bert_batch_size", 64)))
        bert_lr = str(float(config.get("bert_lr", 2e-5)))
        bert_max_len = str(int(config.get("bert_max_len", 128)))
        bert_auto_tune = "1" if bool(config.get("bert_auto_tune", False)) else "0"

        time_test_ratio = str(float(config.get("test_ratio", 0.2)))

        fusion_epochs = str(int(config.get("fusion_epochs", 30)))
        fusion_batch = str(int(config.get("fusion_batch_size", 128)))
        fusion_lr = str(float(config.get("fusion_lr", 5e-3)))
        fusion_test_ratio = str(float(config.get("test_ratio", 0.2)))

        env.update(
            {
                "CSV_PATH": str(work_csv_path),
                "IMG_DIR": str(work_dir / "images"),
                "TEST_RATIO": time_test_ratio,
                "VIT_EPOCHS": vit_epochs,
                "VIT_BATCH_SIZE": vit_batch,
                "VIT_LR": vit_lr,
                "VIT_AUTO_TUNE": vit_auto_tune,
                "BERT_EPOCHS": bert_epochs,
                "BERT_BATCH_SIZE": bert_batch,
                "BERT_LR": bert_lr,
                "BERT_MAX_LEN": bert_max_len,
                "BERT_AUTO_TUNE": bert_auto_tune,
                "TIME_TEST_RATIO": time_test_ratio,
                "FUSION_EPOCHS": fusion_epochs,
                "FUSION_BATCH_SIZE": fusion_batch,
                "FUSION_LR": fusion_lr,
                "FUSION_TEST_RATIO": fusion_test_ratio,
            }
        )

        _set_job(job_id, progress=5, step="vit")
        _run_script(job_id, VIT_SCRIPT, work_dir, env, "ViTtraining")

        vit_out = work_dir / "vit_finetune_out" / "predictions_id_aggregated.csv"
        if not vit_out.is_file():
            raise FileNotFoundError("ViT: vit_finetune_out/predictions_id_aggregated.csv")
        shutil.copy2(vit_out, work_dir / "predictions_id_aggregated.csv")

        _set_job(job_id, progress=45, step="bert")
        _run_script(job_id, BERT_SCRIPT, work_dir, env, "BERTtraining")

        bert_out = work_dir / "cbert_binary_gpu_out" / "predictions_binary_with_fit.csv"
        if not bert_out.is_file():
            raise FileNotFoundError("BERT: cbert_binary_gpu_out/predictions_binary_with_fit.csv")
        shutil.copy2(bert_out, work_dir / "predictions_binary_with_fit.csv")

        _set_job(job_id, progress=70, step="time")
        _run_script(job_id, TIME_SCRIPT, work_dir, env, "time")

        time_out = work_dir / "time_fit_no_image" / "predictions_with_time_only.csv"
        if not time_out.is_file():
            raise FileNotFoundError("time: time_fit_no_image/predictions_with_time_only.csv")
        shutil.copy2(time_out, work_dir / "predictions_with_time_only.csv")

        _set_job(job_id, progress=80, step="fusion")
        _run_script(job_id, FUSION_SCRIPT, work_dir, env, "fusiontraining")

        fusion_result = work_dir / "fusion_out" / "result.csv"
        if not fusion_result.is_file():
            raise FileNotFoundError("fusion: fusion_out/result.csv")

        _set_job(job_id, progress=95, step="finalize")
        work_df, work_err = _read_csv_validate(work_csv_path)
        if work_err:
            raise RuntimeError(work_err)
        rows = _predict_with_project_fusion(work_df, work_dir / "images", fusion_csv_path=fusion_result)
        result_id = _now_tag()
        _save_result(result_id, rows)

        _set_job(job_id, status="completed", progress=100, step="done", result_id=result_id)
    except Exception as e:
        _append_job_log(job_id, f"[ERROR] {e}")
        _set_job(job_id, status="failed", progress=100, step="failed", error=str(e))



def _detect_gpu() -> Tuple[bool, Optional[str]]:
    try:
        import torch
        import sys
        # check CUDA available
        if not torch.cuda.is_available():
            # reason
            cuda_version = torch.version.cuda or "unknown"
            pytorch_version = torch.__version__
            reason = f"CUDAavailable (torch={pytorch_version}, cuda={cuda_version})"
            return False, reason
        # device
        device_count = torch.cuda.device_count()
        current_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_idx)
        cuda_version = torch.version.cuda
        # returndevice
        info = f"{device_name} (CUDA:{cuda_version}, devices:{device_count})"
        return True, info
    except Exception as e:
        return False, f"detecterror: {e}"


def _read_csv_validate(csv_path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    encodings = [
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
    ]
    df = None
    used_encoding = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return pd.DataFrame(), f"CSVreadfailed: {e}"
    if df is None:
        return pd.DataFrame(), "CSVencoding， UTF-8  GBK encoding"
    if df.shape[0] == 0:
        return pd.DataFrame(), "CSVis empty"
    if df.shape[1] < 7:
        return pd.DataFrame(), "CSVinsufficient columns，7column（1columnID、2columntime、4columntext、7columnlabel）"
    # optional：encoding（）
    # print(f"[INFO] CSV encoding detected: {used_encoding}")
    return df, None


def _extract_zip(zip_path: Path, out_dir: Path) -> Tuple[int, Optional[str]]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad:
                return 0, f"zip archivefile: {bad}"

            def is_unsafe(member: str) -> bool:
                p = Path(member)
                if p.is_absolute():
                    return True
                return ".." in p.parts

            unsafe = [m for m in zf.namelist() if is_unsafe(m)]
            if unsafe:
                return 0, "zip archive（）"

            zf.extractall(out_dir)
    except zipfile.BadZipFile:
        return 0, "zip archiveZIPfile"
    except Exception as e:
        return 0, f"failed: {e}"

    cnt = 0
    for p in out_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
            cnt += 1
    return cnt, None


def _collect_images_by_id(images_root: Path) -> Dict[str, List[Path]]:
    id2paths: Dict[str, List[Path]] = {}
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_IMAGE_EXTS:
            continue
        stem = p.stem
        base_id = stem.split("_")[0] if "_" in stem else stem
        if not base_id:
            continue
        id2paths.setdefault(base_id, []).append(p)
    for k in list(id2paths.keys()):
        id2paths[k] = sorted(id2paths[k])
    return id2paths


_FUSION_CACHE: Optional[pd.DataFrame] = None


def _load_fusion_df(fusion_csv_path: Optional[Path] = None) -> pd.DataFrame:
    global _FUSION_CACHE
    if fusion_csv_path is None and _FUSION_CACHE is not None:
        return _FUSION_CACHE

    path = fusion_csv_path or FUSION_RESULT_PATH
    if not path.is_file():
        raise FileNotFoundError(
            "not foundfusionfile: 3 in 1/fusion_out/result.csv。"
            "fusionresult（filedirectory）。"
        )

    df = None
    for _enc in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
        try:
            df = pd.read_csv(path, encoding=_enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise RuntimeError("fusionresultCSVencoding， UTF-8/UTF-8-SIG  GBK encoding")
    if df.empty:
        raise ValueError("fusionresultfileis empty")

    if "id" not in df.columns:
        df.insert(0, "id", df.iloc[:, 0].astype(str))
    df["id"] = df["id"].astype(str)

    if fusion_csv_path is None:
        _FUSION_CACHE = df
    return df


def _predict_with_project_fusion(df: pd.DataFrame, images_root: Path, fusion_csv_path: Optional[Path] = None) -> List[Dict]:
    fusion = _load_fusion_df(fusion_csv_path)
    id_series = df.iloc[:, 0].astype(str)
    want_ids = set(id_series.tolist())

    fusion_ids = set(fusion["id"].astype(str).tolist())
    hit = len(want_ids & fusion_ids)
    if hit == 0:
        raise ValueError(
            "uploadCSVIDfusionresult(result.csv)。"
            "uploaddata，datafusionresult。"
        )

    id2imgs = _collect_images_by_id(images_root)
    fusion_map = fusion.set_index("id")

    def pick_num(row: pd.Series, keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in row.index:
                v = row[k]
                if pd.isna(v):
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    def _series_mean(keys: List[str]) -> Optional[float]:
        for k in keys:
            if k in fusion.columns:
                s = pd.to_numeric(fusion[k], errors="coerce")
                s = s[np.isfinite(s)]
                if len(s) > 0:
                    return float(s.mean())
        return None

    prior_fused = _series_mean(["fused_prob", "fusion_prob", "prob", "p_fused"]) 
    prior_img = _series_mean(["img_prob", "image_prob", "p_img", "vit_prob"])
    prior_txt = _series_mean(["txt_prob", "text_prob", "p_txt", "bert_prob"])
    prior_time = _series_mean(["time_prob", "p_time"])
    if prior_fused is None:
        prior_fused = np.nanmean([x for x in [prior_img, prior_txt, prior_time] if x is not None]) if any(
            x is not None for x in [prior_img, prior_txt, prior_time]
        ) else 0.5
    if prior_img is None:
        prior_img = float(prior_fused)
    if prior_txt is None:
        prior_txt = float(prior_fused)
    if prior_time is None:
        prior_time = float(prior_fused)

    def norm_prob(p: Optional[float], default: float) -> float:
        if p is None:
            return float(default)
        try:
            x = float(p)
        except Exception:
            return float(default)
        if not np.isfinite(x):
            return float(default)
        return float(min(1.0, max(0.0, x)))

    out: List[Dict] = []
    for sid in id_series:
        if sid in fusion_map.index:
            r = fusion_map.loc[sid]
            img_prob = pick_num(r, ["img_prob", "image_prob", "p_img", "vit_prob"])
            img_prob_neg = pick_num(r, ["img_prob_neg", "image_prob_neg", "p_img_neg", "vit_prob_neg"])
            txt_prob = pick_num(r, ["txt_prob", "text_prob", "p_txt", "bert_prob"])
            txt_prob_neg = pick_num(r, ["txt_prob_neg", "text_prob_neg", "p_txt_neg", "bert_prob_neg"])
            time_prob = pick_num(r, ["time_prob", "p_time"])
            time_prob_neg = pick_num(r, ["time_prob_neg", "p_time_neg"])
            fused_prob = pick_num(r, ["fused_prob", "fusion_prob", "prob", "p_fused"])
            fused_prob_neg = pick_num(r, ["fused_prob_neg", "fusion_prob_neg", "prob_neg", "p_fused_neg", "negative"])
            fused_pred = pick_num(r, ["fused_pred", "fusion_pred", "pred"])

            attn_img = pick_num(r, ["attn_img", "w_img", "weight_img"])
            attn_txt = pick_num(r, ["attn_txt", "w_txt", "weight_txt"])
            attn_time = pick_num(r, ["attn_time", "w_time", "weight_time"])
        else:
            img_prob = None
            img_prob_neg = None
            txt_prob = None
            txt_prob_neg = None
            time_prob = None
            time_prob_neg = None
            fused_prob = None
            fused_prob_neg = None
            fused_pred = None
            attn_img = None
            attn_txt = None
            attn_time = None

        # Fill missing values using available modalities + global priors, not fixed constants
        if img_prob is None:
            if txt_prob is not None and time_prob is not None:
                img_prob = (float(txt_prob) + float(time_prob)) / 2.0
            elif txt_prob is not None:
                img_prob = float(txt_prob)
            elif time_prob is not None:
                img_prob = float(time_prob)
            else:
                img_prob = prior_img

        if txt_prob is None:
            if img_prob is not None and time_prob is not None:
                txt_prob = (float(img_prob) + float(time_prob)) / 2.0
            elif img_prob is not None:
                txt_prob = float(img_prob)
            elif time_prob is not None:
                txt_prob = float(time_prob)
            else:
                txt_prob = prior_txt

        if time_prob is None:
            if img_prob is not None and txt_prob is not None:
                time_prob = (float(img_prob) + float(txt_prob)) / 2.0
            elif img_prob is not None:
                time_prob = float(img_prob)
            elif txt_prob is not None:
                time_prob = float(txt_prob)
            else:
                time_prob = prior_time

        # If current ID has no images, derive image probability from text/time (not fixed value)
        if sid not in id2imgs:
            img_prob = (float(txt_prob) + float(time_prob)) / 2.0

        # If only negative probability available, derive positive probability first
        if img_prob is None and img_prob_neg is not None:
            img_prob = 1.0 - float(img_prob_neg)
        if txt_prob is None and txt_prob_neg is not None:
            txt_prob = 1.0 - float(txt_prob_neg)
        if time_prob is None and time_prob_neg is not None:
            time_prob = 1.0 - float(time_prob_neg)
        if fused_prob is None and fused_prob_neg is not None:
            fused_prob = 1.0 - float(fused_prob_neg)

        img_prob = norm_prob(img_prob, prior_img)
        txt_prob = norm_prob(txt_prob, prior_txt)
        time_prob = norm_prob(time_prob, prior_time)
        if fused_prob is None:
            fused_prob = (img_prob + txt_prob + time_prob) / 3.0
        fused_prob = norm_prob(fused_prob, prior_fused)

        if fused_pred is not None:
            fused_pred = int(round(float(fused_pred)))
        else:
            fused_pred = int(fused_prob >= 0.5)

        # Keep existing negative probability if available; otherwise derive from positive probability
        img_prob_neg = norm_prob(img_prob_neg, 1.0 - img_prob) if img_prob_neg is not None else (1.0 - img_prob)
        txt_prob_neg = norm_prob(txt_prob_neg, 1.0 - txt_prob) if txt_prob_neg is not None else (1.0 - txt_prob)
        time_prob_neg = norm_prob(time_prob_neg, 1.0 - time_prob) if time_prob_neg is not None else (1.0 - time_prob)
        fused_prob_neg = norm_prob(fused_prob_neg, 1.0 - fused_prob) if fused_prob_neg is not None else (1.0 - fused_prob)

        out.append(
            {
                "id": sid,
                "img_prob": img_prob,
                "img_prob_neg": img_prob_neg,
                "txt_prob": txt_prob,
                "txt_prob_neg": txt_prob_neg,
                "time_prob": time_prob,
                "time_prob_neg": time_prob_neg,
                "fused_prob": fused_prob,
                "fused_prob_neg": fused_prob_neg,
                "fused_pred": fused_pred,
                "positive": fused_prob,
                "negative": fused_prob_neg,
                "attn_img": attn_img,
                "attn_txt": attn_txt,
                "attn_time": attn_time,
            }
        )
    return out


def _save_result(result_id: str, rows: List[Dict]) -> Path:
    result_dir = RESULTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "result.json"
    csv_path = result_dir / "result.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    return result_dir


def _load_result_rows(result_id: str) -> Optional[List[Dict]]:
    json_path = RESULTS_DIR / result_id / "result.json"
    if not json_path.is_file():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _page_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Fusion System</title>
  <link rel="stylesheet" href="/static/css/style.css" />
</head>
<body>
  <div class="container">
    <header>
      <h1>🚀 Fusion System</h1>
      <p class="subtitle">Upload images (ZIP) + CSV to get fusion results</p>
    </header>

    <div class="status-bar" id="statusBar">
      <div class="status-item">
        <span class="status-label">System:</span>
        <span class="status-value" id="systemStatus">Checking...</span>
      </div>
      <div class="status-item">
        <span class="status-label">GPU:</span>
        <span class="status-value" id="gpuStatus">-</span>
      </div>
    </div>

    <main>
      <div class="card">
        <h2>📁 1: Upload Files</h2>

        <div class="config-section" style="margin-top: 0">
          <div class="config-item">
            <label>Images ZIP archive</label>
            <input id="zipInput" type="file" accept=".zip" />
          </div>
          <div class="config-item">
            <label>CSV File</label>
            <input id="csvInput" type="file" accept=".csv" />
          </div>
        </div>

        <div style="margin-top: 12px; color: var(--text-secondary); font-size: 0.95rem">
          <div>CSV: 7 columns; col1=ID, col2=time ("0101 23:08"), col4=text, col7=label (-1/0/1)</div>
          <div>ZIP archive: images named by ID (e.g., 123.jpg, 123_0.png)</div>
        </div>
      </div>

      <div class="card">
        <h2>⚙️ 2: Training Config</h2>
        <div class="config-section" style="margin-top: 0">
          <div class="config-item">
            <label>Test Ratio</label>
            <input id="testRatio" type="number" min="0.05" max="0.5" step="0.05" value="0.2" />
          </div>
          <div class="config-item">
            <label>ViT epochs</label>
            <input id="vitEpochs" type="number" min="1" max="50" value="3" />
          </div>
          <div class="config-item">
            <label>ViT batch size</label>
            <input id="vitBatch" type="number" min="1" max="256" value="32" />
          </div>
          <div class="config-item">
            <label>ViT lr</label>
            <input id="vitLr" type="number" step="0.000001" value="0.00003" />
          </div>

          <div class="config-item">
            <label>BERT epochs</label>
            <input id="bertEpochs" type="number" min="1" max="20" value="2" />
          </div>
          <div class="config-item">
            <label>BERT batch size</label>
            <input id="bertBatch" type="number" min="1" max="256" value="64" />
          </div>
          <div class="config-item">
            <label>ViT Auto-Tune Runtime</label>
            <select id="vitAutoTune">
              <option value="1">Enabled</option>
              <option value="0" selected>Disabled</option>
            </select>
          </div>
          <div class="config-item">
            <label>BERT Auto-Tune Runtime</label>
            <select id="bertAutoTune">
              <option value="1">Enabled</option>
              <option value="0" selected>Disabled</option>
            </select>
          </div>
          <div class="config-item">
            <label>BERT Max Length</label>
            <input id="bertMaxLen" type="number" min="32" max="256" value="128" />
          </div>
          <div class="config-item">
            <label>BERT lr</label>
            <input id="bertLr" type="number" step="0.000001" value="0.00002" />
          </div>

          <div class="config-item">
            <label>Fusion epochs</label>
            <input id="fusionEpochs" type="number" min="1" max="200" value="30" />
          </div>
          <div class="config-item">
            <label>Fusion batch size</label>
            <input id="fusionBatch" type="number" min="1" max="2048" value="128" />
          </div>
          <div class="config-item">
            <label>Fusion lr</label>
            <input id="fusionLr" type="number" step="0.0001" value="0.005" />
          </div>
        </div>
      </div>

      <div class="card">
        <h2>🎯 3: Run Training</h2>
        <button class="btn-primary" id="predictBtn">Run Training</button>

        <div class="progress-section" id="progressSection" style="display:none">
          <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
          <div class="progress-text" id="progressText">Preparing...</div>
        </div>

        <div style="margin-top: 12px;">
          <textarea id="jobLogs" style="width:100%; height:180px; display:none; padding:12px; border:1px solid var(--border-color); border-radius:8px; background: var(--bg-color);" readonly></textarea>
        </div>
      </div>

      <div class="card" id="resultsCard" style="display:none">
        <h2>📊 Results</h2>
        <div class="results-summary" id="resultsSummary"></div>
        <div class="results-table-container">
          <table class="results-table" id="resultsTable">
            <thead>
              <tr>
                <th>ID</th>
                <th>Image Probability</th>
                <th>Text Probability</th>
                <th>Time Probability</th>
                <th>Fusion Probability</th>
                <th>Prediction</th>
                <th>Weights</th>
              </tr>
            </thead>
            <tbody id="resultsBody"></tbody>
          </table>
        </div>
        <div class="results-actions">
          <button class="btn-secondary" id="downloadBtn">📥 Download CSV</button>
          <button class="btn-secondary" id="exportBtn">📄 Export JSON</button>
        </div>
      </div>
    </main>

    <footer>
      <p>Copyright Fusion System | Powered by Flask</p>
      <p>© Fusion System | Powered by Flask</p>
    </footer>
  </div>

<script>
const API_BASE = '';
let currentResultId = null;
let currentUploadId = null;
let currentJobId = null;

function setStatus(ok, gpuOk, gpuName) {
  const systemStatusEl = document.getElementById('systemStatus');
  const gpuStatusEl = document.getElementById('gpuStatus');
  systemStatusEl.textContent = ok ? 'OK' : 'Error';
  systemStatusEl.classList.remove('success', 'error');
  systemStatusEl.classList.add(ok ? 'success' : 'error');
  if (gpuOk) {
    gpuStatusEl.textContent = `✓ GPU Mode (${gpuName || 'GPU Available'})`;
    gpuStatusEl.classList.remove('error');
    gpuStatusEl.classList.add('success');
  } else {
    gpuStatusEl.textContent = '✗ CPU Mode';
    gpuStatusEl.classList.remove('success');
    gpuStatusEl.classList.add('error');
  }
}

async function checkSystemStatus() {
  try {
    const resp = await fetch(`${API_BASE}/api/health`);
    const data = await resp.json();
    setStatus(data.status === 'ok', !!data.gpu_available, data.gpu_name);
  } catch (e) {
    setStatus(false, false, null);
  }
}

function setProgress(pct, text, isError) {
  const sec = document.getElementById('progressSection');
  const fill = document.getElementById('progressFill');
  const txt = document.getElementById('progressText');
  sec.style.display = 'block';
  fill.style.width = `${pct}%`;
  txt.textContent = text;
  fill.style.background = isError ? 'var(--error-color)' : 'var(--primary-color)';
}

function displayResults(results) {
  const summary = document.getElementById('resultsSummary');
  summary.innerHTML = `
    <div class="summary-item"><div class="summary-label">Total</div><div class="summary-value">${results.length}</div></div>
    <div class="summary-item"><div class="summary-label">Images</div><div class="summary-value">${results.filter(r => r.img_prob !== null && r.img_prob !== undefined).length}</div></div>
    <div class="summary-item"><div class="summary-label">Fusion</div><div class="summary-value">${results.filter(r => r.fused_prob !== null && r.fused_prob !== undefined).length}</div></div>
  `;

  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = results.map(row => {
    const fmtProb = (pPos, pNeg) => {
      const toNum = (x) => {
        if (x === null || x === undefined) return null;
        const n = Number(x);
        return Number.isFinite(n) ? n : null;
      };
      const posRaw = toNum(pPos);
      const negRaw = toNum(pNeg);
      let pos = posRaw;
      let neg = negRaw;
      if (pos === null && neg !== null) pos = 1 - neg;
      if (neg === null && pos !== null) neg = 1 - pos;
      if (pos === null || neg === null) return '-';
      pos = Math.min(1, Math.max(0, pos));
      neg = Math.min(1, Math.max(0, neg));
      return `+${pos.toFixed(4)} / -${neg.toFixed(4)}`;
    };
    const fusedPred = row.fused_pred;
    const predClass = fusedPred === 1 ? 'positive' : 'negative';
    const predText = fusedPred === 1 ? 'positive' : 'negative';
    const hasAttn = (row.attn_img !== null && row.attn_img !== undefined &&
                     row.attn_txt !== null && row.attn_txt !== undefined &&
                     row.attn_time !== null && row.attn_time !== undefined);
    const attnHtml = hasAttn ? `
      <div class="attn-weights">
        <span class="attn-item" title="Image weight">Img: ${(row.attn_img * 100).toFixed(1)}%</span>
        <span class="attn-item" title="Text weight">Txt: ${(row.attn_txt * 100).toFixed(1)}%</span>
        <span class="attn-item" title="Time weight">Time: ${(row.attn_time * 100).toFixed(1)}%</span>
      </div>
    ` : '-';
    return `
      <tr>
        <td>${row.id}</td>
        <td>${fmtProb(row.img_prob, row.img_prob_neg)}</td>
        <td>${fmtProb(row.txt_prob, row.txt_prob_neg)}</td>
        <td>${fmtProb(row.time_prob, row.time_prob_neg)}</td>
        <td>${fmtProb(row.fused_prob, row.fused_prob_neg ?? row.negative)}</td>
        <td><span class="pred-badge ${predClass}">${predText}</span></td>
        <td>${attnHtml}</td>
      </tr>
    `;
  }).join('');

  document.getElementById('resultsCard').style.display = 'block';
  document.getElementById('resultsCard').scrollIntoView({behavior: 'smooth'});
}

async function loadResults(resultId) {
  const resp = await fetch(`${API_BASE}/api/results/${resultId}`);
  const data = await resp.json();
  if (data.status !== 'success') {
    throw new Error(data.error || 'Failed to load results');
  }
  displayResults(data.data);
}

async function startPredict() {
  const zipFile = document.getElementById('zipInput').files[0];
  const csvFile = document.getElementById('csvInput').files[0];
  if (!zipFile || !csvFile) {
    alert('Please select both images ZIP archive and CSV file');
    return;
  }
  if (!zipFile.name.toLowerCase().endsWith('.zip')) {
    alert('Please upload a ZIP archive for images');
    return;
  }
  if (!csvFile.name.toLowerCase().endsWith('.csv')) {
    alert('Please upload a CSV file');
    return;
  }

  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span><span>Processing...</span>';

  try {
    setProgress(15, 'Uploading files...');
    const fd = new FormData();
    fd.append('images_archive', zipFile);
    fd.append('csv_file', csvFile);

    const resp = await fetch(`${API_BASE}/api/upload_bundle`, { method: 'POST', body: fd });
    const up = await resp.json();
    if (up.status !== 'success') {
      throw new Error(up.error || 'Upload failed');
    }

    currentUploadId = up.upload_id;

    const cfg = {
      test_ratio: Number(document.getElementById('testRatio').value || 0.2),
      vit_epochs: Number(document.getElementById('vitEpochs').value || 3),
      vit_batch_size: Number(document.getElementById('vitBatch').value || 32),
      vit_lr: Number(document.getElementById('vitLr').value || 0.00003),
      bert_epochs: Number(document.getElementById('bertEpochs').value || 2),
      bert_batch_size: Number(document.getElementById('bertBatch').value || 64),
      bert_max_len: Number(document.getElementById('bertMaxLen').value || 128),
      bert_lr: Number(document.getElementById('bertLr').value || 0.00002),
      vit_auto_tune: document.getElementById('vitAutoTune').value === '1',
      bert_auto_tune: document.getElementById('bertAutoTune').value === '1',
      fusion_epochs: Number(document.getElementById('fusionEpochs').value || 30),
      fusion_batch_size: Number(document.getElementById('fusionBatch').value || 128),
      fusion_lr: Number(document.getElementById('fusionLr').value || 0.005),
    };

    setProgress(25, 'Creating training job...');
    const startResp = await fetch(`${API_BASE}/api/start_training`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: currentUploadId, config: cfg })
    });
    const sd = await startResp.json();
    if (sd.status !== 'success') {
      throw new Error(sd.error || 'Failed to start training');
    }
    currentJobId = sd.job_id;

    const logBox = document.getElementById('jobLogs');
    logBox.style.display = 'block';
    logBox.value = '';

    setProgress(30, 'Training in progress...');

    while (true) {
      await new Promise(r => setTimeout(r, 2000));
      const jr = await fetch(`${API_BASE}/api/job/${currentJobId}`);
      const jd = await jr.json();
      if (jd.status !== 'success') {
        throw new Error(jd.error || 'Job failed');
      }

      const pct = jd.job.progress || 0;
      const step = jd.job.step || '';
      setProgress(Math.min(99, pct), `Training: ${step} (${pct}%)`);
      const logs = jd.job.logs || [];
      logBox.value = logs.join('\\n');
      logBox.scrollTop = logBox.scrollHeight;

      if (jd.job.status === 'completed') {
        currentResultId = jd.job.result_id;
        break;
      }
      if (jd.job.status === 'failed') {
        throw new Error(jd.job.error || 'Training failed');
      }
    }

    setProgress(100, 'Training completed, loading results...');
    await loadResults(currentResultId);
  } catch (e) {
    setProgress(100, `failed: ${e.message}`, true);
    alert(e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">▶</span><span>Run Training</span>';
  }
}

document.getElementById('predictBtn').addEventListener('click', startPredict);
document.getElementById('downloadBtn').addEventListener('click', () => {
  if (!currentResultId) return alert('No results to download');
  window.open(`${API_BASE}/api/download/${currentResultId}`, '_blank');
});
document.getElementById('exportBtn').addEventListener('click', async () => {
  if (!currentResultId) return alert('No results to export');
  const resp = await fetch(`${API_BASE}/api/results/${currentResultId}`);
  const data = await resp.json();
  if (data.status !== 'success') return alert(data.error || 'Export failed');
  const jsonStr = JSON.stringify(data.data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `predictions_${currentResultId}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

checkSystemStatus();
</script>
</body>
</html>"""


app = Flask(__name__, static_folder=str(APP_ROOT / "static"), static_url_path="/static")
CORS(app)


@app.get("/")
def index():
    return Response(_page_html(), mimetype="text/html; charset=utf-8")


@app.get("/favicon.ico")
def favicon():
    return Response(status=204)


@app.get("/api/health")
def health():
    gpu_ok, gpu_name = _detect_gpu()
    return jsonify({"status": "ok", "gpu_available": gpu_ok, "gpu_name": gpu_name})


@app.post("/api/upload_bundle")
def upload_bundle():
    _ensure_dirs()

    if "images_archive" not in request.files or "csv_file" not in request.files:
        return _json_error("Please upload both images_archive (ZIP) and csv_file (CSV)")

    images_file = request.files.get("images_archive")
    csv_file = request.files.get("csv_file")

    if not images_file or not images_file.filename:
        return _json_error("No images ZIP archive provided")
    if not csv_file or not csv_file.filename:
        return _json_error("No CSV file provided")

    img_name = _safe_filename(images_file.filename)
    csv_name = _safe_filename(csv_file.filename)
    img_ext = Path(img_name).suffix.lower()
    csv_ext = Path(csv_name).suffix.lower()

    if img_ext not in ALLOWED_ARCHIVE_EXTS:
        return _json_error("Images must be a ZIP archive")
    if csv_ext not in ALLOWED_CSV_EXTS:
        return _json_error("CSV file must have .csv extension")

    upload_id = _now_tag()
    upload_dir = UPLOADS_DIR / upload_id
    images_dir = upload_dir / "images"
    upload_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    zip_path = upload_dir / img_name
    csv_path = upload_dir / csv_name

    try:
        images_file.save(zip_path)
        csv_file.save(csv_path)
    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return _json_error(f"File upload failed: {e}")

    df, err = _read_csv_validate(csv_path)
    if err:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return _json_error(err)

    img_count, zerr = _extract_zip(zip_path, images_dir)
    if zerr:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return _json_error(zerr)
    if img_count <= 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return _json_error("ZIP archive contains no image files")

    ids = set(df.iloc[:, 0].astype(str).tolist())
    id2imgs = _collect_images_by_id(images_dir)
    hit = sum(1 for i in ids if i in id2imgs)
    if hit == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return _json_error(
            "No matching images found in ZIP archive for CSV IDs. Images should be named by ID (e.g., 123.jpg, 123_0.png)",
            details={"csv_id_samples": list(sorted(list(ids)))[:10]},
        )

    return jsonify(
        {
            "status": "success",
            "upload_id": upload_id,
            "csv_path": str(csv_path),
            "images_dir": str(images_dir),
            "image_count": img_count,
            "matched_id_count": hit,
            "total_id_count": len(ids),
        }
    )


@app.post("/api/start_training")
def start_training():
    _ensure_dirs()
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    config = payload.get("config") or {}
    if not upload_id:
        return _json_error("upload_id is required")

    job_id = _now_tag()
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "upload_id": str(upload_id),
            "status": "pending",
            "progress": 0,
            "step": "queued",
            "result_id": None,
            "error": None,
            "logs": [],
            "created_at": datetime.now().isoformat(),
            "config": config,
        }

    t = threading.Thread(target=_train_pipeline, args=(job_id, str(upload_id), config), daemon=True)
    t.start()
    return jsonify({"status": "success", "job_id": job_id})


@app.get("/api/job/<job_id>")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return _json_error("job_id does not exist", 404)
        return jsonify({"status": "success", "job": job})


@app.post("/api/predict_bundle")
def predict_bundle():
    _ensure_dirs()
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    if not upload_id:
        return _json_error("upload_id is required")

    upload_dir = UPLOADS_DIR / str(upload_id)
    if not upload_dir.is_dir():
        return _json_error("upload_id does not exist or was cleaned")

    images_dir = upload_dir / "images"
    if not images_dir.is_dir():
        return _json_error("Images directory not found")

    csv_files = [p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    if not csv_files:
        return _json_error("No CSV file found in upload")
    csv_path = csv_files[0]

    df, err = _read_csv_validate(csv_path)
    if err:
        return _json_error(err)

    try:
        rows = _predict_with_project_fusion(df, images_dir)
    except FileNotFoundError as e:
        return _json_error(str(e), 500)
    except ValueError as e:
        return _json_error(str(e), 400)
    except Exception as e:
        return _json_error(f"Server error: {e}", 500)

    result_id = _now_tag()
    _save_result(result_id, rows)
    return jsonify({"status": "success", "result_id": result_id})


@app.get("/api/results/<result_id>")
def get_results(result_id: str):
    rows = _load_result_rows(result_id)
    if rows is None:
        return _json_error("Result does not exist", 404)
    return jsonify({"status": "success", "data": rows})


@app.get("/api/download/<result_id>")
def download_result(result_id: str):
    csv_path = RESULTS_DIR / result_id / "result.csv"
    if not csv_path.is_file():
        return _json_error("Result file does not exist", 404)
    return send_file(
        csv_path,
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name=f"predictions_{result_id}.csv",
    )


@app.post("/api/cleanup")
def cleanup():
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    if not upload_id:
        return _json_error("upload_id is required")
    upload_dir = UPLOADS_DIR / str(upload_id)
    if not upload_dir.exists():
        return jsonify({"status": "success", "deleted": False})
    try:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"status": "success", "deleted": True})
    except Exception as e:
        return _json_error(f"Cleanup failed: {e}", 500)


if __name__ == "__main__":
    _ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
