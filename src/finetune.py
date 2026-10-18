"""cross-encoder の out-of-fold ファインチューニング（リークを防ぐため、LightGBM の CV と同じ fold で学習する）"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.embedding import GPU_PYTHON

FINETUNE_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "finetune_ce.py"


def fold_ids(groups: pd.Series, n_folds: int = 5) -> np.ndarray:
    """src/train.py の run_cv_lgb と同じ GroupKFold の分け方で、各行の fold 番号を返す（分け方は groups だけで決まる）"""
    folds = np.full(len(groups), -1)
    for k, (_, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups)):
        folds[va_idx] = k
    return folds


def finetune_ce_oof(train_query: pd.Series, train_doc: pd.Series, relevance: np.ndarray, groups: pd.Series,
                    test_query: pd.Series, test_doc: pd.Series, out_dir: Path, model: str,
                    n_folds: int = 5, epochs: float = 2, logger=None) -> tuple[np.ndarray, np.ndarray]:
    """fold ごとに cross-encoder をファインチューニングし、(train の OOF スコア, test のスコア（fold の平均）) を返す。
    fold ごとの結果は out_dir に保存し、保存済みの fold は学習し直さない"""
    folds = fold_ids(groups, n_folds)
    labels = (relevance - 1) / 2  # relevance（1〜3）を 0〜1 に変換する
    test_keys = (test_query + "\x1f" + test_doc)
    test_uniq = test_keys.drop_duplicates()
    data = {
        "train": [[q, d, float(l), int(f)] for q, d, l, f in zip(train_query, train_doc, labels, folds)],
        "test": [k.split("\x1f", 1) for k in test_uniq],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / "input.json"
    data_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    for k in range(n_folds):
        cmd = [str(GPU_PYTHON), str(FINETUNE_SCRIPT), "--data", str(data_path), "--fold", str(k),
               "--out-dir", str(out_dir), "--model", model, "--epochs", str(epochs)]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace") as proc:
            for line in proc.stdout:
                line = line.rstrip()
                # 進捗の行だけを残し、ライブラリの警告などは捨てる
                if logger and line.startswith(("step ", "fold ", "device=")):
                    logger.info(f"finetune: {line}")
        if proc.returncode != 0:
            raise RuntimeError(f"finetune_ce.py failed on fold {k} (exit {proc.returncode})")

    oof = np.zeros(len(train_query), dtype=np.float32)
    test_scores = np.zeros(len(test_uniq), dtype=np.float32)
    for k in range(n_folds):
        oof[folds == k] = np.load(out_dir / f"oof_fold{k}.npy")
        test_scores += np.load(out_dir / f"test_fold{k}.npy") / n_folds
    test_map = dict(zip(test_uniq, test_scores))
    return oof, test_keys.map(test_map).to_numpy(dtype=np.float32)
