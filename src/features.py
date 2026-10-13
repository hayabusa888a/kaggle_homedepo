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


def tfidf_cosine_features(df: pd.DataFrame) -> pd.DataFrame:
    """train + test を結合した df を受け取り、検索語と商品テキストの TF-IDF コサイン類似度を返す"""
    vectorizer = TfidfVectorizer()  # 既定値: 小文字化あり、2文字以上の単語、語彙の絞り込みなし
    vectorizer.fit(build_corpus(df))

    query_vec = vectorizer.transform(df["search_term"])
    title_vec = vectorizer.transform(df["product_title"])
    desc_vec = vectorizer.transform(df["product_description"])

    return pd.DataFrame(
        {
            "cos_title": rowwise_cosine(query_vec, title_vec),
            "cos_desc": rowwise_cosine(query_vec, desc_vec),
        },
        index=df.index,
    )
