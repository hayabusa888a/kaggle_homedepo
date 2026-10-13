import pandas as pd

from src.utils import INPUT_DIR

ENCODING = "ISO-8859-1"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """train / test を読み込み、商品説明文を product_uid で結合して返す"""
    train = pd.read_csv(INPUT_DIR / "train.csv", encoding=ENCODING)
    test = pd.read_csv(INPUT_DIR / "test.csv", encoding=ENCODING)
    desc = pd.read_csv(INPUT_DIR / "product_descriptions.csv", encoding=ENCODING)

    train = train.merge(desc, on="product_uid", how="left")
    test = test.merge(desc, on="product_uid", how="left")
    return train, test
