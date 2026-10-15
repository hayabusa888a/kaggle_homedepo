import re
from collections import Counter

import pandas as pd
from symspellpy import SymSpell, Verbosity

TOKEN_RE = re.compile(r"\w+")


def build_symspell(product_texts: pd.Series, min_count: int = 2, max_edit_distance: int = 2) -> SymSpell:
    """商品テキスト（正規化後）の語彙を辞書にした SymSpell を作る。頻度は「その単語を含む商品テキストの数」"""
    counts = Counter()
    for text in product_texts:
        counts.update(set(TOKEN_RE.findall(text)))
    sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
    for word, c in counts.items():
        if c >= min_count and not any(ch.isdigit() for ch in word):
            sym.create_dictionary_entry(word, c)
    return sym


RARE_COUNT = 5  # 辞書にあってもこの回数未満の単語は、誤字の可能性を疑う
RARE_RATIO = 50  # 編集距離1の候補がこの倍以上よく出てくれば、そちらに直す


def _best_suggestion(sym: SymSpell, word: str, max_dist: int):
    """編集距離が最小の候補のうち、先頭の文字が同じものを優先し、その中で最頻の候補を返す"""
    suggestions = [s for s in sym.lookup(word, Verbosity.ALL, max_edit_distance=max_dist) if s.term != word]
    if not suggestions:
        return None
    min_dist = min(s.distance for s in suggestions)
    suggestions = [s for s in suggestions if s.distance == min_dist]
    same_head = [s for s in suggestions if s.term[0] == word[0]]
    candidates = same_head or suggestions
    return max(candidates, key=lambda s: s.count) if candidates else None


def correct_word(sym: SymSpell, word: str) -> str:
    """単語を 連結語の分割 → 編集距離の近い単語 の順に修正する。直せなければそのまま返す"""
    if len(word) < 3 or any(ch.isdigit() for ch in word):
        return word

    count = sym.words.get(word, 0)
    if count >= RARE_COUNT:
        return word
    if count > 0:
        # 辞書にあるが珍しい単語: 編集距離1に圧倒的によく出る単語があれば誤字とみなす ("flourescent" -> "fluorescent")
        best = _best_suggestion(sym, word, max_dist=1)
        if best is not None and best.term != word and best.count >= count * RARE_RATIO:
            return best.term
        return word

    # 候補1: 編集距離の近い単語（短い単語は誤修正しやすいので、許す編集距離を小さくする）
    best = _best_suggestion(sym, word, max_dist=1 if len(word) <= 4 else 2)
    # 候補2: 連結語の分割 "airconditioner" -> "air conditioner"（分割しすぎを防ぐため、各語3文字以上・3語まで）
    parts = sym.word_segmentation(word, max_edit_distance=0).corrected_string.split()
    seg_ok = 2 <= len(parts) <= 3 and all(len(p) >= 3 and p in sym.words for p in parts)

    # 両方あれば出現回数で比べる。分割は「一番珍しい部分」の回数で評価する
    # ("dreamline": dream line > dreamlike、"refridgerator": refrigerator > ref ridg erator)
    if seg_ok and (best is None or min(sym.words[p] for p in parts) > best.count):
        return " ".join(parts)
    return best.term if best is not None else word


def make_query_corrector(sym: SymSpell):
    """検索語の Series を受け取り、単語ごとにスペル修正した Series を返す関数を作る"""
    cache: dict[str, str] = {}

    def fix(word: str) -> str:
        if word not in cache:
            cache[word] = correct_word(sym, word)
        return cache[word]

    def correct(queries: pd.Series) -> pd.Series:
        uniq = queries.drop_duplicates()
        fixed = uniq.map(lambda q: TOKEN_RE.sub(lambda m: fix(m.group()), q))
        return queries.map(dict(zip(uniq, fixed)))

    correct.cache = cache
    return correct
