"""ローカル LLM（Ollama）による検索語のスペル修正"""
import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.preprocess import _map_unique, normalize

OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = """You are a spelling corrector for search queries typed by customers on the
Home Depot website (home improvement products: tools, hardware, building
materials, appliances, plumbing, electrical, paint, garden, etc.).

Fix only spelling mistakes in each query, including misspelled brand names
and words that were joined together by mistake (e.g. "airconditioner" -> "air conditioner").
Rules:
- Do not add, remove, or reorder words, except to split joined words.
- Keep correctly spelled words, numbers, sizes, and model numbers exactly as they are.
- Do not explain or comment. Write all output in lowercase.
- If a query has no mistakes, return it unchanged.
Return every query with its original id."""

# 構造化出力: この形の JSON 以外は生成されない
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "query": {"type": "string"}},
                "required": ["id", "query"],
            },
        }
    },
    "required": ["results"],
}


def find_unknown_queries(df: pd.DataFrame) -> list[str]:
    """正規化後に「商品テキストに一度も出てこない単語（3文字以上・数字なし）」を含む検索語（元の文字列）を返す"""
    products = df.drop_duplicates("product_uid")
    product_texts = _map_unique(products["product_title"] + " " + products["product_description"], normalize)
    vocab = set()
    for text in product_texts:
        vocab.update(re.findall(r"\w+", text))

    def has_unknown(query: str) -> bool:
        return any(
            len(w) >= 3 and not any(ch.isdigit() for ch in w) and w not in vocab
            for w in re.findall(r"\w+", normalize(query))
        )

    queries = df["search_term"].drop_duplicates()
    return queries[queries.map(has_unknown)].tolist()


def _request(model: str, queries: list[str]) -> dict[int, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                {"queries": [{"id": i, "query": q} for i, q in enumerate(queries)]}, ensure_ascii=False)},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as res:
        content = json.load(res)["message"]["content"]
    return {r["id"]: r["query"] for r in json.loads(content)["results"]}


def correct_queries(queries: list[str], cache_path: Path, model: str = "llama3.1:8b",
                    batch_size: int = 30, logger=None, log_every: int = 10) -> dict[str, str]:
    """検索語をまとめて LLM に送り、{元の検索語: 修正後} を返す。結果は cache_path に逐次保存し、保存済みの検索語は送らない。
    logger を渡すと log_every バッチごとに途中経過（件数、経過時間、残り時間の目安、修正例）を出す"""
    cache: dict[str, str] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [q for q in queries if q not in cache]
    if logger:
        logger.info(f"llm spell: {len(queries)} queries, {len(queries) - len(todo)} cached, {len(todo)} to process")
    n_fallback = 0
    t0 = time.time()
    n_batches = (len(todo) + batch_size - 1) // batch_size
    for b, start in enumerate(range(0, len(todo), batch_size), 1):
        batch = todo[start : start + batch_size]
        result = _request(model, batch)
        if any(i not in result for i in range(len(batch))):
            result = _request(model, batch)  # ID が欠けたら1回だけ再送する
        for i, q in enumerate(batch):
            if i in result:
                cache[q] = result[i]
            else:
                cache[q] = q  # それでも欠けたら元の検索語のまま
                n_fallback += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        if logger and (b % log_every == 0 or b == n_batches):
            done, elapsed = min(start + batch_size, len(todo)), time.time() - t0
            eta = elapsed / done * (len(todo) - done)
            changed = [(q, cache[q]) for q in batch if cache[q].lower() != q.lower()][:3]
            logger.info(f"llm spell: {done}/{len(todo)} ({done / len(todo):.0%}) elapsed {elapsed / 60:.1f}min "
                        f"eta {eta / 60:.1f}min fallback {n_fallback} | e.g. {changed}")
    return {q: cache[q] for q in queries}
