# Project 1 : Content-based Recommendation
# 2025-27567 홍길동

import os
import json
import hashlib
from collections import Counter
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
RATINGS_DATA = f"{DATA_DIR}/ratings.csv"
METADATA_DATA = f"{DATA_DIR}/games_metadata.json"
RECOMMEND_INPUT = "input_recommendation.txt"       # 4-b: user_id 목록
PREDICTION_INPUT = "input_score_prediction.txt"    # 4-c: user_id;game_id 목록
OUTPUT_DIR = "results"
CACHE_DIR = "cache"                                # 재실행 속도용 캐시 (채점에 불필요)
SEMANTIC_CACHE = f"{CACHE_DIR}/semantic_matrix.npz"
SEMANTIC_MODEL = "all-MiniLM-L6-v2"
TOP_K = 20
EPS = 1e-9                                         # 유사도 합이 0일 때 0 나눗셈 방지
ALPHA = 0.8                                        # validate.py(holdout + 5-fold)로 선택, 0.5 대비 RMSE 개선
RATING_MIN, RATING_MAX = 1.0, 5.0                  # rating 범위 (4-c 예측 clip용)


# game_id(원본 ID, 예: 282440) -> 행렬 row index. 행렬 접근 전 반드시 이 매핑을 거친다
def build_game_index(game_ids: np.ndarray) -> dict[int, int]:
    return {int(game_id): row for row, game_id in enumerate(game_ids)}


# user_id -> (평가한 game의 row index, rating). user x game 행렬(103GiB, 99.99% 빈 칸) 대신
# user_id로 정렬한 테이블에서 요청받은 user의 구간만 꺼내 쓴다
class UserHistory:
    def __init__(self, ratings: pd.DataFrame, game_index: dict[int, int]):
        rows = ratings["game_id"].map(game_index)

        # metadata에 없는 game_id는 representation이 없으므로 제외 (현재 데이터에는 없음)
        if rows.isna().any():
            keep = rows.notna()
            print(f"[warn] dropping {(~keep).sum()} ratings whose game_id is not in metadata")
            ratings, rows = ratings[keep], rows[keep]

        self.table = pd.DataFrame(
            {
                "row": rows.to_numpy(dtype=np.intp),
                "rating": ratings["rating"].to_numpy(dtype=np.float64),
            },
            index=ratings["user_id"].to_numpy(),
        ).sort_index()

    def get(self, user_id: int) -> tuple[np.ndarray, np.ndarray]:
        if user_id not in self.table.index:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float64)
        # 리스트로 감싸야 이력이 1개인 user도 Series가 아닌 DataFrame으로 반환됨
        sub = self.table.loc[[user_id]]
        return sub["row"].to_numpy(), sub["rating"].to_numpy()

    def contains(self, user_ids) -> np.ndarray:
        return np.isin(np.asarray(user_ids), self.table.index.to_numpy())


# Content 기반 Item-based CF. representation만 바꿔 semantics용/TF-IDF용 인스턴스를 각각 만든다
class ItemCF:
    def __init__(self, vectors: np.ndarray, game_ids: np.ndarray,
                 game_index: dict[int, int], history: UserHistory):
        # 행 단위 L2 정규화 -> 이후 내적이 곧 cosine similarity
        V = np.asarray(vectors, dtype=np.float64)
        self.V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)
        self.game_ids = game_ids
        self.game_index = game_index
        self.history = history

    # user의 전체 game 추정 점수 [19755] = sum_j r_j * sim(j, i) / (sum_j sim(j, i) + eps)
    # sim에 선형이므로 [n, 19755] 유사도 행렬 없이 가중합 벡터 하나로 접는다 (수학적으로 동일)
    def predict_all(self, user_id: int) -> np.ndarray:
        rows, ratings = self.history.get(user_id)
        if len(rows) == 0:
            return np.zeros(len(self.V), dtype=np.float64)

        rated = self.V[rows]                            # [n, d]
        numerator = self.V @ (ratings @ rated)          # [19755]
        denominator = self.V @ rated.sum(axis=0)        # [19755]
        return numerator / (denominator + EPS)

    def predict(self, user_id: int, game_id: int) -> float:
        return float(self.predict_all(user_id)[self.game_index[game_id]])

    # 여러 (user_id, game_id) 쌍을 한 번에 예측. 쌍마다 [19755] 전체 대신 필요한 값만 계산
    # (쌍마다 user 이력을 펼쳐 내적한 뒤 bincount로 되합침, 총 sum_u n_u^2 ≈ 309만 번의 내적)
    def predict_pairs(self, user_ids, game_ids, max_rows: int = 50_000) -> np.ndarray:
        targets = np.array([self.game_index[int(g)] for g in game_ids], dtype=np.intp)
        users = np.asarray(user_ids)

        # history는 user_id로 정렬되어 있어 각 쌍의 user 구간을 searchsorted로 한 번에 찾는다
        hist_users = self.history.table.index.to_numpy()
        hist_rows = self.history.table["row"].to_numpy()
        hist_ratings = self.history.table["rating"].to_numpy()
        lo = np.searchsorted(hist_users, users, side="left")
        counts = np.searchsorted(hist_users, users, side="right") - lo

        # 펼친 이력 행 수가 chunk당 max_rows 이하가 되도록 쌍을 나눠 처리 (이력 없는 user는 0으로 남음)
        cum = np.cumsum(counts)
        preds = np.zeros(len(users), dtype=np.float64)
        start = 0
        while start < len(users):
            base = cum[start - 1] if start > 0 else 0
            end = max(start + 1, int(np.searchsorted(cum, base + max_rows, side="right")))
            n = counts[start:end]
            total = int(n.sum())
            if total > 0:
                # pair_of[k] = 펼친 k번째 항목이 속한 쌍의 번호
                pair_of = np.repeat(np.arange(end - start), n)
                within = np.arange(total) - np.repeat(np.cumsum(n) - n, n)   # user 구간 내 offset
                idx = np.repeat(lo[start:end], n) + within

                sims = np.einsum("ij,ij->i", self.V[targets[start:end][pair_of]], self.V[hist_rows[idx]])
                numerator = np.bincount(pair_of, weights=sims * hist_ratings[idx], minlength=end - start)
                denominator = np.bincount(pair_of, weights=sims, minlength=end - start)
                preds[start:end] = numerator / (denominator + EPS)
            start = end

        return preds

    # 미평가 game 중 상위 k개 [(game_id, score), ...]
    # 출력되는 4자리 점수 기준 내림차순, 동점이면 game_id 오름차순 (반올림 전 점수는 float 오차가 순서를 정함)
    def recommend(self, user_id: int, k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.predict_all(user_id)
        rated_rows, _ = self.history.get(user_id)
        candidates = np.setdiff1d(np.arange(len(scores)), rated_rows)

        key = np.round(scores[candidates], 4)
        order = candidates[np.lexsort((self.game_ids[candidates], -key))[:k]]
        return [(int(self.game_ids[row]), float(scores[row])) for row in order]


# 4-c: 두 ItemCF 점수의 alpha 가중합. RMSE로만 평가되므로 [1, 5]로 clip하고,
# 이력 없는 user(두 모델 모두 0)는 평균 rating으로 대체 (4-b 추천에는 미적용)
class WeightedItemCF:
    def __init__(self, semantic: ItemCF, tfidf: ItemCF, alpha: float = ALPHA):
        self.semantic = semantic
        self.tfidf = tfidf
        self.alpha = alpha
        self.history = semantic.history
        self.global_mean = float(self.history.table["rating"].mean())

    def predict_pairs(self, user_ids, game_ids) -> np.ndarray:
        users = np.asarray(user_ids)
        return self.combine(users, self.semantic.predict_pairs(users, game_ids),
                            self.tfidf.predict_pairs(users, game_ids))

    # 가중합 + 후처리. 검증 시 두 모델 점수는 한 번만 구하고 alpha만 바꿔 재사용하도록 분리
    def combine(self, users: np.ndarray, sem_scores: np.ndarray, tfidf_scores: np.ndarray) -> np.ndarray:
        scores = self.alpha * sem_scores + (1 - self.alpha) * tfidf_scores
        scores = np.clip(scores, RATING_MIN, RATING_MAX)
        scores[~self.history.contains(users)] = self.global_mean
        return scores

    def predict(self, user_id: int, game_id: int) -> float:
        return float(self.predict_pairs([user_id], [game_id])[0])

    def score_rmse(self, ratings: pd.DataFrame) -> float:
        # 평가할 ratings가 모델 history에 포함되어 있으면 누수이므로 train/validation을 분리해 쓸 것
        preds = self.predict_pairs(ratings["user_id"].to_numpy(), ratings["game_id"].to_numpy())
        return float(np.sqrt(np.mean((preds - ratings["rating"].to_numpy(dtype=np.float64)) ** 2)))

# games_metadata.json(한 줄에 JSON 하나)을 읽어 game_id순으로 정렬된 game_id, description, tags 반환
def load_game_data()  -> tuple[np.ndarray, list[str], list[list[str]]]:
    metadata = []

    with open(METADATA_DATA, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))

    metadata = sorted(metadata, key=lambda x: x["game_id"])
    game_ids = np.array([m["game_id"] for m in metadata])
    descriptions = [m["description"] for m in metadata]
    tags = [m["tags"] for m in metadata]

    return game_ids, descriptions, tags

def load_ratings_data() -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS_DATA)
    return ratings

# description을 all-MiniLM-L6-v2로 벡터화 [game 수, 384]. 인코딩이 느려 npz로 캐싱하고,
# 모델 이름 + description 전체의 해시가 캐시와 다르면 다시 인코딩
def build_semantic_matrix(descriptions: list[str]) -> np.ndarray:
    key = hashlib.sha256(json.dumps([SEMANTIC_MODEL, descriptions]).encode("utf-8")).hexdigest()
    if os.path.exists(SEMANTIC_CACHE):
        with np.load(SEMANTIC_CACHE) as cached:
            if str(cached["key"]) == key:
                print(f"[cache] reuse {SEMANTIC_CACHE} (shape={cached['embeddings'].shape})")
                return cached["embeddings"]
        print(f"[cache] {SEMANTIC_CACHE} is stale, re-encoding")

    model = SentenceTransformer(SEMANTIC_MODEL)
    embeddings = np.array(model.encode(descriptions, show_progress_bar=True))

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(SEMANTIC_CACHE, embeddings=embeddings, key=key)
    return embeddings

# tag TF-IDF 행렬 [game 수, unique tag 수]. 열은 tag의 alphabetical order
def build_tfidf_matrix(tags: list[list[str]]) -> np.ndarray:
    unique_tags = sorted({tag for game_tags in tags for tag in game_tags})
    tag_col = {tag: i for i, tag in enumerate(unique_tags)}

    # IDF = log10(전체 game 수 / 해당 tag가 등록된 game 수)
    total_count = len(tags)
    tag_counts = Counter()

    for game_tags in tags:
        tag_counts.update(set(game_tags))
    idf = {tag: np.log10(total_count / tag_counts[tag]) for tag in unique_tags}

    # TF = n(d, t) / n(d). 각 game이 가진 tag만 순회하며 해당 열에 누적
    tfidf_matrix = np.zeros((len(tags), len(unique_tags)))
    for row, game_tags in enumerate(tags):
        n_d = len(game_tags)
        if n_d == 0:
            continue
        for tag in game_tags:
            tfidf_matrix[row, tag_col[tag]] += idf[tag] / n_d

    return tfidf_matrix

# input_recommendation.txt: 한 줄에 user_id 하나
def read_recommendation_input() -> list[int]:
    with open(RECOMMEND_INPUT, "r", encoding="utf-8") as f:
        return [int(l.strip()) for l in f.readlines() if l.strip()]


# input_score_prediction.txt: 한 줄에 "user_id;game_id"
def read_prediction_input() -> list[tuple[int, int]]:
    pairs = []
    with open(PREDICTION_INPUT, "r", encoding="utf-8") as f:
        for l in f.readlines():
            l = l.strip()
            if not l:
                continue
            uid, gid = l.split(";")
            pairs.append((int(uid), int(gid)))
    return pairs


# newline="\n": Windows에서도 CRLF가 아닌 LF로 기록해 OS와 무관하게 같은 파일을 만든다
def write_output(lines: list[str], filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")

# 출력 한 줄 = "user_id,game_id,score" (공백 없는 콤마, 소수점 4자리 반올림)

# 4-b: user당 상위 TOP_K개. semantics()/tfidf()는 representation만 다르고 출력 로직은 같다
def recommend_lines(user_ids: list[int], recommendor: ItemCF) -> list[str]:
    result = []
    for user_id in user_ids:
        for game_id, score in recommendor.recommend(user_id, TOP_K):
            result.append("{},{},{:.4f}".format(user_id, game_id, score))
    return result

def semantics(user_ids: list[int], recommendor: ItemCF) -> list[str]:
    return recommend_lines(user_ids, recommendor)

def tfidf(user_ids: list[int], recommendor: ItemCF) -> list[str]:
    return recommend_lines(user_ids, recommendor)

def weighted(pairs: list[tuple[int, int]], recommendor: WeightedItemCF) -> list[str]:
    result = []
    user_ids = [user_id for user_id, _ in pairs]
    game_ids = [game_id for _, game_id in pairs]
    scores = recommendor.predict_pairs(user_ids, game_ids)
    for user_id, game_id, score in zip(user_ids, game_ids, scores):
        result.append("{},{},{:.4f}".format(user_id, game_id, score))
    return result


if __name__ == "__main__":
    game_ids, descriptions, tags = load_game_data()
    ratings = load_ratings_data()
    game_index = build_game_index(game_ids)
    history = UserHistory(ratings, game_index)
    semantic_matrix = build_semantic_matrix(descriptions)
    tfidf_matrix = build_tfidf_matrix(tags)
    semantic_recommendor = ItemCF(semantic_matrix, game_ids, game_index, history)
    tfidf_recommendor = ItemCF(tfidf_matrix, game_ids, game_index, history)
    weighted_recommendor = WeightedItemCF(semantic_recommendor, tfidf_recommendor, ALPHA)

    user_ids = read_recommendation_input()
    result_sem = semantics(user_ids, semantic_recommendor)
    result_tfidf = tfidf(user_ids, tfidf_recommendor)

    pairs = read_prediction_input()
    result_weighted = weighted(pairs, weighted_recommendor)

    write_output(result_sem, "semantics_output.txt")
    write_output(result_tfidf, "tfidf_output.txt")
    write_output(result_weighted, "score_prediction_output.txt")
