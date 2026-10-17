"""exp016: exp014 の特徴量で LightGBM のハイパーパラメータを Optuna で調整する"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging

import numpy as np
import optuna
import pandas as pd
from scipy import sparse

from src.data import load_attribute_texts, load_brands, load_data
from src.embedding import embed_texts_gpu
from src.rerank import rerank_pairs_gpu
from src.features import (attr_match_features, bm25_features, brand_features, char_cosine_features,
                          cosine_features, match_features, tfidf_vectors)
from src.preprocess import _map_unique, normalize, preprocess, stem
from src.spell import build_symspell, make_query_corrector
from src.train import run_cv_lgb
from src.utils import ROOT, OUTPUT_DIR, get_logger, rmse, seed_everything, timer

EXP_NAME = "exp016"
SEED = 42
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "seed": SEED,
    "verbose": -1,
}
# 2文字以上の単語 + 1文字の数字（寸法 "2 x 4" の数字を残すため）
TOKEN_PATTERN = r"(?u)\b(?:\w\w+|\d)\b"
N_TRIALS = 30
SEARCH_LR = 0.1  # 探索中は learning_rate を上げて 1 回の試行を速くする
FINAL_LR = 0.02  # 最終的なモデルは learning_rate を下げて学習する
EMB_DIR = ROOT / "data" / "processed" / "emb_mxbai"
EMB_DIM = 1024
EMB_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
# mxbai-embed-large の決まり: 検索語にだけ目印を付け、文書側には付けない
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"
RERANK_DIR = ROOT / "data" / "processed" / "rerank_mxbai_base"


def main():
    out_dir = OUTPUT_DIR / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(out_dir / "log.txt")
    seed_everything(SEED)

    with timer("load data", logger):
        train, test = load_data()
        df = pd.concat([train, test], ignore_index=True)
        # ハイフンは正規化の「連結形も残す」処理を通さず、単純にスペースにする ("Strong-Tie" -> "strong tie")
        raw_brands = load_brands()
        brands = raw_brands.str.replace("-", " ").map(lambda b: stem(normalize(b)))

    with timer("spell dictionary", logger):
        products = df.drop_duplicates("product_uid")
        product_texts = _map_unique(products["product_title"] + " " + products["product_description"], normalize)
        corrector = make_query_corrector(build_symspell(product_texts))

    with timer("embeddings", logger):
        # 埋め込みにはステミング前の自然な文を使う。検索語は正規化 + SymSpell 後（ステミング前）、商品側は元のテキスト
        query_text = corrector(_map_unique(df["search_term"], normalize))
        emb_q = embed_texts_gpu(query_text, QUERY_PREFIX, EMB_DIR / "query.npz", EMB_MODEL, logger=logger)
        emb_t = embed_texts_gpu(df["product_title"], "", EMB_DIR / "title.npz", EMB_MODEL, logger=logger)
        emb_d = embed_texts_gpu(df["product_description"], "", EMB_DIR / "desc.npz", EMB_MODEL, batch_size=32,
                                logger=logger)
        brand_raw = df["product_uid"].map(raw_brands)
        has_brand = brand_raw.notna().to_numpy()
        emb_b = embed_texts_gpu(brand_raw[has_brand], "", EMB_DIR / "brand.npz", EMB_MODEL, logger=logger)
        emb = pd.DataFrame({
            "emb_cos_title": (emb_q * emb_t).sum(axis=1),
            "emb_cos_desc": (emb_q * emb_d).sum(axis=1),
            "emb_cos_brand": np.nan,
        }, index=df.index)
        emb.loc[has_brand, "emb_cos_brand"] = (emb_q[has_brand] * emb_b).sum(axis=1)

    with timer("rerank", logger):
        # 埋め込みと同じく、検索語は正規化 + SymSpell 後（ステミング前）、商品側は元のテキストを使う
        ce = pd.DataFrame({
            "ce_title": rerank_pairs_gpu(query_text, df["product_title"], RERANK_DIR / "title.npz", RERANK_MODEL, logger=logger),
            "ce_desc": rerank_pairs_gpu(query_text, df["product_description"], RERANK_DIR / "desc.npz", RERANK_MODEL,
                                        batch_size=32, logger=logger),
        }, index=df.index)
        del emb_b

    with timer("preprocess", logger):
        df = preprocess(df, query_fn=corrector)
        attr_text = load_attribute_texts()
        df["attr_text"] = _map_unique(df["product_uid"].map(attr_text).fillna(""), lambda s: stem(normalize(s)))

    with timer("tfidf features", logger):
        vecs, vocab = tfidf_vectors(df, token_pattern=TOKEN_PATTERN)
        cos = cosine_features(vecs, df.index)

    with timer("match features", logger):
        match = match_features(df)

    with timer("brand features", logger):
        brand = brand_features(df, brands)

    with timer("bm25 features", logger):
        bm25 = bm25_features(df, {"title": "product_title", "desc": "product_description", "attr": "attr_text"},
                             token_pattern=TOKEN_PATTERN)

    with timer("char ngram features", logger):
        char = char_cosine_features(df)

    with timer("attr match features", logger):
        attr = attr_match_features(df)

    dense = pd.concat([cos, match, brand, bm25, char, attr, emb, ce], axis=1)
    # 埋め込みベクトル（検索語・タイトル、各 1024 次元）は密な行列だが、TF-IDF ベクトルと横に並べるため疎行列にする。
    # 全体を並べてから train/test に分けるとコピーが重なりメモリが足りなくなるので、先に分けてから並べる
    blocks = [dense.to_numpy(dtype=np.float32), emb_q, emb_t,
              vecs["query"].astype(np.float32), vecs["title"].astype(np.float32)]
    n = len(train)
    X = sparse.hstack([sparse.csr_matrix(b[:n]) for b in blocks], format="csr")
    X_test = sparse.hstack([sparse.csr_matrix(b[n:]) for b in blocks], format="csr")
    del blocks, emb_q, emb_t, emb_d
    feature_names = (list(dense.columns) + [f"eq_{i}" for i in range(EMB_DIM)] + [f"et_{i}" for i in range(EMB_DIM)]
                     + [f"q_{w}" for w in vocab] + [f"t_{w}" for w in vocab])
    y = train["relevance"].to_numpy()
    logger.info(f"vocab size: {len(vocab)}, n_features: {X.shape[1]} (dense {dense.shape[1]})")

    # 追加した特徴量の値ごとの平均 relevance（train）
    dense_train = dense.iloc[: len(train)].assign(relevance=y)
    for col in []:
        stats = dense_train.groupby(col)["relevance"].agg(["mean", "count"]).round(3)
        logger.info(f"relevance by {col}:\n{stats.to_string()}")
    new_cols = list(ce.columns) + list(emb.columns)
    logger.info("new feature stats (train):\n" + dense.iloc[: len(train)][new_cols].describe().T.round(3).to_string())
    corr = dense_train[new_cols + ["cos_title", "ratio_match_title"]].corrwith(dense_train["relevance"])
    logger.info("corr with relevance (train):\n" + corr.round(3).to_string())

    groups = train["search_term"]
    # 試行中の fold ごとのログは出さず、試行ごとの結果だけを出す
    quiet = logging.getLogger("quiet")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False

    def objective(trial: optuna.Trial) -> float:
        params = {
            **LGB_PARAMS,
            "learning_rate": SEARCH_LR,
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        # 探索中は test の予測が不要なので、1 行だけ渡して時間を節約する
        oof_t, _, _ = run_cv_lgb(X, y, X_test[:1], groups, params, quiet, early_stopping_rounds=50)
        score = rmse(y, oof_t)
        logger.info(f"trial {trial.number}: rmse={score:.5f} params={trial.params}")
        return score

    with timer("optuna", logger):
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        # 1 回目は exp014 と同じ既定のパラメータ（learning_rate だけ探索用）にして、比較の基準にする
        study.enqueue_trial({"num_leaves": 31, "min_child_samples": 20, "feature_fraction": 1.0,
                             "bagging_fraction": 1.0, "lambda_l1": 1e-3, "lambda_l2": 1e-3})
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(out_dir / "optuna_trials.csv", index=False)
    logger.info(f"best trial {study.best_trial.number}: rmse={study.best_value:.5f} params={study.best_params}")
    logger.info(f"baseline trial 0 (default params, lr={SEARCH_LR}): rmse={study.trials[0].value:.5f}")

    # 参考: 既定のパラメータのまま learning_rate だけ下げた場合（改善の内訳を分けるため）
    with timer("final default params", logger):
        logger.info(f"--- default params, lr={FINAL_LR}")
        run_cv_lgb(X, y, X_test, groups, {**LGB_PARAMS, "learning_rate": FINAL_LR}, logger)

    best_params = {**LGB_PARAMS, **study.best_params, "bagging_freq": 1, "learning_rate": FINAL_LR}
    logger.info(f"--- best params, lr={FINAL_LR}: {best_params}")
    with timer("final best params", logger):
        oof, pred_test, importance = run_cv_lgb(X, y, X_test, groups, best_params, logger)

    imp = pd.Series(importance, index=feature_names)
    logger.info(f"features used (gain > 0): {(imp > 0).sum()}")
    for prefix, name in [("eq_", "query emb"), ("et_", "title emb"), ("q_", "query tfidf"), ("t_", "title tfidf")]:
        part = imp[imp.index.str.startswith(prefix)]
        logger.info(f"{name}: used {(part > 0).sum()} columns, total gain {part.sum():.1f}")
    logger.info("feature importance top 40 (gain):\n" + imp.sort_values(ascending=False).head(40).round(1).to_string())

    pd.DataFrame({"id": train["id"], "relevance": y, "pred": oof}).to_csv(out_dir / "oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "relevance": pred_test}).to_csv(out_dir / "submission.csv", index=False)
    logger.info(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
