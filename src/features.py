from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """IDF 計算用のコーパス: ユニークな検索語 + 商品ごとの「タイトル + 説明文」"""
    queries = df["search_term"].drop_duplicates()
    products = df.drop_duplicates("product_uid")
    product_docs = products["product_title"] + " " + products["product_description"]
    return pd.concat([queries, product_docs], ignore_index=True)


def rowwise_cosine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    # TfidfVectorizer の出力は行ごとに L2 正規化済みなので、内積 = コサイン類似度
    return np.asarray(a.multiply(b).sum(axis=1)).ravel()


def tfidf_vectors(df: pd.DataFrame, **vectorizer_kwargs) -> tuple[dict[str, sparse.csr_matrix], np.ndarray]:
    """train + test を結合した df を受け取り、検索語・タイトル・説明文の TF-IDF ベクトルと語彙を返す"""
    # 既定値: 小文字化あり、2文字以上の単語、語彙の絞り込みなし
    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    vectorizer.fit(build_corpus(df))
    vecs = {
        "query": vectorizer.transform(df["search_term"]),
        "title": vectorizer.transform(df["product_title"]),
        "desc": vectorizer.transform(df["product_description"]),
    }
    return vecs, vectorizer.get_feature_names_out()


def cosine_features(vecs: dict[str, sparse.csr_matrix], index: pd.Index) -> pd.DataFrame:
    """検索語と商品テキストの TF-IDF コサイン類似度"""
    return pd.DataFrame(
        {
            "cos_title": rowwise_cosine(vecs["query"], vecs["title"]),
            "cos_desc": rowwise_cosine(vecs["query"], vecs["desc"]),
        },
        index=index,
    )


def tfidf_cosine_features(df: pd.DataFrame) -> pd.DataFrame:
    vecs, _ = tfidf_vectors(df)
    return cosine_features(vecs, df.index)


def _tokens(s: str) -> list[str]:
    # TF-IDF と同じく、1文字の英字は除き、1文字の数字は残す
    return [w for w in s.split() if len(w) >= 2 or w.isdigit()]


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(phrase) and f" {phrase} " in f" {text} "


def match_features(df: pd.DataFrame) -> pd.DataFrame:
    """前処理済みの df から、検索語と商品テキストの単語一致の特徴量を作る"""
    rows = []
    for q, t, d in zip(df["search_term"], df["product_title"], df["product_description"]):
        qw, tw, dw = _tokens(q), _tokens(t), _tokens(d)
        ts, ds = set(tw), set(dw)
        nq = max(len(qw), 1)
        q_bi = list(zip(qw, qw[1:]))
        t_bi = set(zip(tw, tw[1:]))
        q_num = [w for w in qw if w.isdigit()]
        t_num = {w for w in tw if w.isdigit()}
        n_title = sum(w in ts for w in qw)
        n_desc = sum(w in ds for w in qw)
        rows.append({
            "q_n_words": len(qw),
            "q_n_chars": len(q),
            "t_n_words": len(tw),
            "d_n_words": len(dw),
            "n_match_title": n_title,
            "ratio_match_title": n_title / nq,
            "n_match_desc": n_desc,
            "ratio_match_desc": n_desc / nq,
            "n_match_any": sum(w in ts or w in ds for w in qw),
            "q_in_title": int(_contains_phrase(" ".join(tw), " ".join(qw))),
            "q_in_desc": int(_contains_phrase(" ".join(dw), " ".join(qw))),
            "first_in_title": int(bool(qw) and qw[0] in ts),
            "last_in_title": int(bool(qw) and qw[-1] in ts),
            "last_in_desc": int(bool(qw) and qw[-1] in ds),
            "n_bigram_title": sum(b in t_bi for b in q_bi),
            "ratio_bigram_title": sum(b in t_bi for b in q_bi) / max(len(q_bi), 1),
            "q_n_numbers": len(q_num),
            "ratio_number_title": sum(w in t_num for w in q_num) / len(q_num) if q_num else np.nan,
        })
    return pd.DataFrame(rows, index=df.index)


def brand_features(df: pd.DataFrame, product_brand: pd.Series, mention_purity: float = 0.5) -> pd.DataFrame:
    """前処理済みの df と、前処理済みのブランド名（product_uid -> brand）から、ブランド一致の特徴量を作る。

    検索語に「別のブランド」が含まれるかを見るため、ブランド名の一覧を作る。ただし "home" や "light" のような
    一般語のブランド名を拾わないように、そのフレーズをタイトルに含む商品のうち mention_purity 以上が
    そのブランドの商品であるものだけを「ブランドとして言及されている」とみなす"""
    brand = df["product_uid"].map(product_brand)
    products = df.drop_duplicates("product_uid")
    p_brand = products["product_uid"].map(product_brand)

    # 各ブランドについて「タイトルにそのフレーズを含む商品数」と「そのうち同ブランドの商品数」を数える
    candidates = {b for b in p_brand.dropna().unique() if 1 <= len(b.split()) <= 4}
    with_phrase, same_brand = Counter(), Counter()
    for title, b in zip(products["product_title"], p_brand):
        words = title.split()
        grams = {" ".join(words[i : i + n]) for n in range(1, 5) for i in range(len(words) - n + 1)}
        for g in grams & candidates:
            with_phrase[g] += 1
            same_brand[g] += int(g == b)
    known = {b for b in candidates if with_phrase[b] and same_brand[b] / with_phrase[b] >= mention_purity}

    def mentioned(query: str) -> set[str]:
        words = query.split()
        grams = {" ".join(words[i : i + n]) for n in range(1, 5) for i in range(len(words) - n + 1)}
        return grams & known

    rows = []
    for q, b in zip(df["search_term"], brand):
        m = mentioned(q)
        has_brand = isinstance(b, str)
        rows.append({
            "has_brand": int(has_brand),
            "brand_in_query": int(has_brand and _contains_phrase(q, b)) if has_brand else np.nan,
            "query_mentions_brand": int(bool(m)),
            "other_brand_in_query": int(has_brand and any(x != b and not _contains_phrase(b, x) for x in m)),
        })
    return pd.DataFrame(rows, index=df.index)


def bm25_features(df: pd.DataFrame, fields: dict[str, str], token_pattern: str,
                  k1: float = 1.2, b: float = 0.75) -> pd.DataFrame:
    """検索語と各フィールドの BM25 スコア。IDF と平均文書長は商品単位（product_uid でユニーク）で計算する。
    fields は {特徴量名の接尾辞: 列名}"""
    from sklearn.feature_extraction.text import CountVectorizer

    out = {}
    products = df.drop_duplicates("product_uid")
    for name, col in fields.items():
        cv = CountVectorizer(token_pattern=token_pattern)
        docs_p = cv.fit_transform(products[col])
        n_docs = docs_p.shape[0]
        doc_freq = np.asarray((docs_p > 0).sum(axis=0)).ravel()
        idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        avgdl = docs_p.sum(axis=1).mean()

        tf = cv.transform(df[col]).tocsr().astype(np.float64)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        # 非ゼロ要素ごとに tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)) * idf を計算する
        row_norm = k1 * (1 - b + b * dl / avgdl)
        rows = np.repeat(np.arange(tf.shape[0]), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + row_norm[rows]) * idf[tf.indices]

        query = cv.transform(df["search_term"])
        query.data[:] = 1  # 検索語の単語は重複を数えない
        out[f"bm25_{name}"] = np.asarray(query.multiply(tf).sum(axis=1)).ravel()
        # 検索語の単語の IDF の合計で割った値（検索語の長さの影響を除いた、0〜k1+1 の範囲のスコア）
        q_idf = np.asarray(query.multiply(idf[np.newaxis, :]).sum(axis=1)).ravel()
        out[f"bm25_{name}_norm"] = np.divide(out[f"bm25_{name}"], q_idf, out=np.zeros_like(q_idf), where=q_idf > 0)
    return pd.DataFrame(out, index=df.index)


def char_cosine_features(df: pd.DataFrame, ngram_range: tuple[int, int] = (3, 5)) -> pd.DataFrame:
    """検索語とタイトルの、文字 n-gram の TF-IDF コサイン類似度（残ったスペルミスや表記揺れに強い）"""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, min_df=2, sublinear_tf=True)
    vectorizer.fit(pd.concat([df["search_term"].drop_duplicates(), df["product_title"].drop_duplicates()]))
    return pd.DataFrame(
        {"char_cos_title": rowwise_cosine(vectorizer.transform(df["search_term"]),
                                          vectorizer.transform(df["product_title"]))},
        index=df.index,
    )


def attr_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """前処理済みの attr_text 列（商品の属性値の連結）と検索語の単語一致"""
    rows = []
    for q, a in zip(df["search_term"], df["attr_text"]):
        qw, aset = _tokens(q), set(_tokens(a))
        n = sum(w in aset for w in qw)
        rows.append({"n_match_attr": n, "ratio_match_attr": n / max(len(qw), 1), "has_attr": int(bool(aset))})
    return pd.DataFrame(rows, index=df.index)
