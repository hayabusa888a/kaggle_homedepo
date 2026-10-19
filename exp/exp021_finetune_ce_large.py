"""exp021: exp020 + より大きいモデル（bge-reranker-v2-m3）でファインチューニングした cross-encoder（検索語 × タイトル）のスコア"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import load_attribute_texts, load_brands, load_data
from src.embedding import embed_texts_gpu
from src.finetune import finetune_ce_oof
from src.rerank import rerank_pairs_gpu
from src.features import (attr_match_features, group_relative_features, bm25_features, brand_features, char_cosine_features,
                          cosine_features, match_features, tfidf_vectors)
from src.preprocess import _map_unique, normalize, preprocess, stem
from src.spell import build_symspell, make_query_corrector
from src.train import run_cv_lgb
from src.utils import ROOT, OUTPUT_DIR, get_logger, rmse, seed_everything, timer

EXP_NAME = "exp021"
SEED = 42
# exp016 で調整したパラメータ（learning_rate 0.02）
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 66,
    "feature_fraction": 0.3136763356113579,
    "bagging_fraction": 0.9252144229403493,
    "bagging_freq": 1,
    "lambda_l1": 9.04067224246189,
    "lambda_l2": 0.004600545339775671,
    "seed": SEED,
    "verbose": -1,
}
# 同じ検索語の中で比べる特徴量（重要度の高いもの）
GROUP_COLS = ["ft_ce_large_title", "ft_ce_title_desc", "ft_ce_title", "ce_title", "ce_desc", "emb_cos_title", "emb_cos_desc", "ratio_match_title", "cos_title",
              "bm25_title_norm", "char_cos_title"]
# 2文字以上の単語 + 1文字の数字（寸法 "2 x 4" の数字を残すため）
TOKEN_PATTERN = r"(?u)\b(?:\w\w+|\d)\b"
EMB_DIR = ROOT / "data" / "processed" / "emb_mxbai"
EMB_DIM = 1024
EMB_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
# mxbai-embed-large の決まり: 検索語にだけ目印を付け、文書側には付けない
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"
RERANK_DIR = ROOT / "data" / "processed" / "rerank_mxbai_base"
# mxbai-rerank-base-v1（DeBERTa-v3）は今の環境では学習が NaN になるため、XLM-RoBERTa 系の bge-reranker-base を使う
FT_MODEL = "BAAI/bge-reranker-base"
FT_DIR = ROOT / "data" / "processed" / "ft_ce_bge_base_title"
FT_EPOCHS = 2
FT_TD_DIR = ROOT / "data" / "processed" / "ft_ce_bge_base_title_desc"
FT_TD_MAX_LENGTH = 256  # タイトル + 説明文の先頭。長い分は説明文の後ろから切り捨てる
# 大きいモデル（5.7 億パラメータ）。8GB の GPU に収めるため、単語の埋め込み層（約半分の 2.65 億）を固定し、8bit 版の AdamW を使う
FT_LARGE_MODEL = "BAAI/bge-reranker-v2-m3"
FT_LARGE_DIR = ROOT / "data" / "processed" / "ft_ce_bge_v2m3_title"
FT_LARGE_ARGS = ["--freeze-embeddings", "--optim", "adamw_bnb_8bit"]


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

    raw_query = df["search_term"].copy()  # グループ化のキー（前処理前の元の検索語）
    df_raw_title = df["product_title"].copy()  # ファインチューニングの入力（前処理前のタイトル）
    df_raw_desc = df["product_description"].copy()

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

    with timer("finetune cross-encoder", logger):
        # LightGBM の CV と同じ fold（検索語単位の GroupKFold）で学習し、学習に使っていない行のスコアだけを使う
        n = len(train)
        ft_oof, ft_test = finetune_ce_oof(query_text[:n], df_raw_title[:n], train["relevance"].to_numpy(),
                                          train["search_term"], query_text[n:], df_raw_title[n:], FT_DIR,
                                          FT_MODEL, epochs=FT_EPOCHS, logger=logger)
        logger.info(f"finetuned cross-encoder standalone oof rmse: {rmse(train['relevance'], ft_oof * 2 + 1):.5f}")

    with timer("finetune cross-encoder (title + description)", logger):
        title_desc = df_raw_title + ". " + df_raw_desc
        td_oof, td_test = finetune_ce_oof(query_text[:n], title_desc[:n], train["relevance"].to_numpy(),
                                          train["search_term"], query_text[n:], title_desc[n:], FT_TD_DIR,
                                          FT_MODEL, epochs=FT_EPOCHS, max_length=FT_TD_MAX_LENGTH, logger=logger)
        logger.info(f"finetuned cross-encoder (title + desc) standalone oof rmse: {rmse(train['relevance'], td_oof * 2 + 1):.5f}")

    with timer("finetune cross-encoder (large, title)", logger):
        lg_oof, lg_test = finetune_ce_oof(query_text[:n], df_raw_title[:n], train["relevance"].to_numpy(),
                                          train["search_term"], query_text[n:], df_raw_title[n:], FT_LARGE_DIR,
                                          FT_LARGE_MODEL, epochs=FT_EPOCHS, extra_args=FT_LARGE_ARGS, logger=logger)
        logger.info(f"finetuned cross-encoder (large, title) standalone oof rmse: {rmse(train['relevance'], lg_oof * 2 + 1):.5f}")
        ft = pd.DataFrame({"ft_ce_title": np.concatenate([ft_oof, ft_test]),
                           "ft_ce_title_desc": np.concatenate([td_oof, td_test]),
                           "ft_ce_large_title": np.concatenate([lg_oof, lg_test])}, index=df.index)

    base = pd.concat([cos, match, brand, bm25, char, attr, emb, ce, ft], axis=1)
    with timer("group relative features", logger):
        # 元の検索語の文字列ごとに、train と test を合わせて集計する（ラベルは使わない）
        grp = group_relative_features(base, raw_query, GROUP_COLS, prefix="q")
    dense = pd.concat([base, grp], axis=1)
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
    new_cols = ["ft_ce_large_title", "ft_ce_title", "ft_ce_title_desc", "q_ft_ce_large_title_diff_mean", "q_ft_ce_large_title_diff_max", "q_ft_ce_large_title_rank"]
    logger.info("new feature stats (train):\n" + dense.iloc[: len(train)][new_cols].describe().T.round(3).to_string())
    corr = dense_train[new_cols + ["cos_title", "ratio_match_title"]].corrwith(dense_train["relevance"])
    logger.info("corr with relevance (train):\n" + corr.round(3).to_string())

    oof, pred_test, importance = run_cv_lgb(X, y, X_test, train["search_term"], LGB_PARAMS, logger)

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
