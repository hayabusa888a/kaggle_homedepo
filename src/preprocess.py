import re
from functools import lru_cache

import pandas as pd
from nltk.stem.snowball import SnowballStemmer

TEXT_COLS = ["search_term", "product_title", "product_description"]

# 数字の直後の単位表記を統一する (例: "12 in." / "36inch" / "8'" -> "12 in" / "36 in" / "8 ft")
UNIT_PATTERNS = [
    (r"(?:square|sq\.?)\s*(?:feet|foot|ft\.?)", "sqft"),
    (r"(?:cubic|cu\.?)\s*(?:feet|foot|ft\.?)", "cuft"),
    (r"(?:inches|inch|in\.|in\b|\")", "in"),
    (r"(?:feet|foot|ft\.|ft\b|')", "ft"),
    (r"(?:gallons|gallon|gal\.|gal\b)", "gal"),
    (r"(?:pounds|pound|lbs\.|lbs\b|lb\.|lb\b)", "lb"),
    (r"(?:ounces|ounce|oz\.|oz\b)", "oz"),
    (r"(?:volts|volt|v\b)", "volt"),
    (r"(?:watts|watt|w\b)", "watt"),
    (r"(?:amps|amp|amperes|ampere)\b", "amp"),
    (r"(?:millimeters|millimeter|mm\b)", "mm"),
    (r"(?:centimeters|centimeter|cm\b)", "cm"),
]
UNIT_PATTERNS = [(re.compile(r"(\d)\s*" + pat), r"\1 " + unit + " ") for pat, unit in UNIT_PATTERNS]

_stemmer = SnowballStemmer("english")


def normalize(s: str) -> str:
    """A. 表記の正規化（連結語の分割、小文字化、ハイフン、寸法、単位）"""
    # 箇条書きの連結を分割: "projectsStronger" -> "projects Stronger"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.lower()
    s = s.replace("&amp;", " ").replace("&nbsp;", " ")
    # 英字同士のハイフンは、分割した形と連結した形の両方を残す: "rust-oleum" -> "rust oleum rustoleum"
    s = re.sub(r"\b([a-z]+)-([a-z]+)\b", r"\1 \2 \1\2", s)
    # 数字の桁区切りを除去: "1,000" -> "1000"
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # 寸法を分解: "2x4x8" -> "2 x 4 x 8"
    s = re.sub(r"(\d)\s*(?:x|\*|by)\s*(?=\d)", r"\1 x ", s)
    for pat, rep in UNIT_PATTERNS:
        s = pat.sub(rep, s)
    return s


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def stem(s: str) -> str:
    """B. ステミング（Snowball）: 単語ごとに語幹へ揃える"""
    return " ".join(_stem(w) for w in re.findall(r"\w+", s))


def preprocess(df: pd.DataFrame, query_fn=None) -> pd.DataFrame:
    """テキスト列に正規化とステミングを適用する。query_fn は検索語だけに挟む処理（スペル修正など）"""
    df = df.copy()
    for col in TEXT_COLS:
        df[col] = _map_unique(df[col], normalize)
    if query_fn is not None:
        df["search_term"] = query_fn(df["search_term"])
    for col in TEXT_COLS:
        df[col] = _map_unique(df[col], stem)
    return df


def _map_unique(s: pd.Series, fn) -> pd.Series:
    # 同じ商品の説明文などが何行も重複するので、ユニークな値だけ処理する
    uniq = s.drop_duplicates()
    return s.map(dict(zip(uniq, uniq.map(fn))))
