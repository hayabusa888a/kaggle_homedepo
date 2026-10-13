# Home Depot Product Search Relevance

Kaggle「[Home Depot Product Search Relevance](https://www.kaggle.com/competitions/home-depot-product-search-relevance)」に取り組むリポジトリ。

- タスク：検索語と商品のペアに対する関連度 `relevance`（1〜3）を予測する
- 評価指標：RMSE

実験の内容と結果は [EXPERIMENTS.md](EXPERIMENTS.md) にまとめている。

## ディレクトリ構成

```
input/          コンペの配布データ（git 管理外）
data/
  processed/    前処理済みデータのキャッシュ（git 管理外）
  features/     特徴量のキャッシュ（git 管理外）
src/
  utils.py      パス定義、ロガー、RMSE、シード固定
  data.py       データ読み込み、商品説明文の結合
  features.py   特徴量の作成（TF-IDF コサイン類似度など）
exp/            実験スクリプト（1実験 = 1ファイル、expXXX_内容.py）
output/expXXX/  実験ごとの出力（log.txt、oof.csv、submission.csv）
notebooks/      EDA 用ノートブック
EXPERIMENTS.md  実験ログ
```

## セットアップ

### 環境

Python 3.14 で動作を確認している。必要なパッケージは次のとおり。

```
pandas numpy scipy scikit-learn lightgbm
```

### データ

コンペのページからデータをダウンロードし、解凍して `input/` に置く。

```
input/
  train.csv
  test.csv
  product_descriptions.csv
  attributes.csv
  sample_submission.csv
```

Kaggle API を使う場合：

```
kaggle competitions download -c home-depot-product-search-relevance -p input
```

## 実験の実行

リポジトリのルートから実行する。

```
python exp/exp001_tfidf_lgbm.py
```

出力は `output/exp001/` に保存される。

| ファイル | 内容 |
|---|---|
| `log.txt` | fold ごとと全体の CV RMSE、特徴量重要度 |
| `oof.csv` | train に対する予測値（`id`, `relevance`, `pred`） |
| `submission.csv` | 提出用ファイル（`id`, `relevance`） |

## 実験の進め方

- CV は GroupKFold（`search_term` 単位、5分割）で統一する。理由は [EXPERIMENTS.md](EXPERIMENTS.md) の「CV の方針」を参照
- 新しい実験は `exp/` に新しいファイルを作り、結果を EXPERIMENTS.md に追記する
