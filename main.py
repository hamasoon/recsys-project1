# Project 1 : Content-based Recommendation
# 2025-27567 홍길동
#
# 제출: main.py + requirements.txt
# 채점: `pip install -r requirements.txt` 후 `python main.py` 로 처음부터 끝까지
#       한 번에 실행됩니다. results/ 폴더에 생성된 output 파일만으로 채점하므로,
#       제출 전 이전에 만든 변수가 하나도 남아있지 않은 새 환경에서
#       이 파일 하나만으로 끝까지 도는지 반드시 확인 후 제출하세요.

# 이 코드가 사용하는 라이브러리는 전부 requirements.txt에 명시되어 있습니다.
# (설치 코드를 main.py에 넣지 마세요 — `!pip install ...`은 python main.py로
#  실행할 때 동작하지 않습니다. 채점 시 `pip install -r requirements.txt`를
#  먼저 실행한 뒤 `python main.py`로 채점합니다.)
# 추가 라이브러리를 쓴다면 import 문을 여기 추가하고, requirements.txt에도 반드시 적으세요.

import os
import json
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
CACHE_DIR = "cache"                                # 재실행 속도용 캐시 (제출/채점에 불필요)
SEMANTIC_CACHE = f"{CACHE_DIR}/semantic_matrix.npy"
TOP_K = 20
EPS = 1e-9                                         # 유사도 합이 0일 때 0으로 나누는 것을 방지
ALPHA = 0.5                                        # TODO: validation으로 최적값을 찾아 교체할 것


# game_id -> representation 행렬의 row index
# ratings의 game_id는 원본 ID(예: 282440)이므로, 행렬에 접근하기 전에 반드시 이 매핑을 거쳐야 함
def build_game_index(game_ids: np.ndarray) -> dict[int, int]:
    return {int(game_id): row for row, game_id in enumerate(game_ids)}


# user_id -> (그 user가 평가한 game의 row index, 그 game에 남긴 rating)
# user 815,006 x game 19,755 행렬은 103GiB이고 99.99%가 빈 칸이므로 만들지 않는다.
# 예측에 필요한 건 user 한 명이 평가한 game(평균 1.3개, 최대 300개)뿐이라,
# user_id로 정렬해두고 요청받은 user의 구간만 꺼내 쓴다.
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
                "rating": ratings["rating"].to_numpy(dtype=np.float32),
            },
            index=ratings["user_id"].to_numpy(),
        ).sort_index()

    def get(self, user_id: int) -> tuple[np.ndarray, np.ndarray]:
        if user_id not in self.table.index:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.float32)
        # 리스트로 감싸야 평가한 game이 1개인 user도 Series가 아닌 DataFrame으로 반환됨
        sub = self.table.loc[[user_id]]
        return sub["row"].to_numpy(), sub["rating"].to_numpy()


# Content 기반 Item-based Collaborative Filtering.
# representation만 갈아끼우면 되므로 semantics용/TF-IDF용 인스턴스를 각각 만들어 쓴다.
class ItemCF:
    def __init__(self, vectors: np.ndarray, game_ids: np.ndarray,
                 game_index: dict[int, int], history: UserHistory):
        # 행 단위 L2 정규화. 이렇게 해두면 이후의 내적이 그대로 cosine similarity가 된다
        V = np.asarray(vectors, dtype=np.float32)
        self.V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)
        self.game_ids = game_ids
        self.game_index = game_index
        self.history = history

    # 특정 user가 전체 game에 남길 것으로 추정되는 점수. shape=[19755]
    #   분자 = sum_j rating_j * sim(j, i)   (평가한 game들의 rating을 유사도로 가중합)
    #   분모 = sum_j sim(j, i) + eps        (유사도 합으로 정규화)
    # 분자/분모 모두 sim에 대해 선형이므로, [n, 19755] 유사도 행렬을 실제로 만들지 않고
    # 가중합 벡터 하나로 접을 수 있다. 근사가 아니라 수학적으로 동일한 값이다.
    def predict_all(self, user_id: int) -> np.ndarray:
        rows, ratings = self.history.get(user_id)
        if len(rows) == 0:
            return np.zeros(len(self.V), dtype=np.float32)

        rated = self.V[rows]                            # [n, d]
        numerator = self.V @ (ratings @ rated)          # [19755]
        denominator = self.V @ rated.sum(axis=0)        # [19755]
        return numerator / (denominator + EPS)

    def predict(self, user_id: int, game_id: int) -> float:
        return float(self.predict_all(user_id)[self.game_index[game_id]])

    # 여러 (user_id, game_id) 쌍을 한 번에 예측. predict()를 반복 호출한 것과 값이 같지만,
    # 쌍마다 [19755] 전체 점수를 만드는 대신 필요한 row 하나만 계산한다.
    #   predict()  : 쌍당 [19755, d] 행렬곱 -> 1.06M쌍이면 사실상 끝나지 않음
    #   여기       : 쌍당 (그 user가 평가한 game 수)번의 내적 -> 전체 합 sum_u n_u^2 = 309만
    # 분자/분모 모두 sim에 대해 선형이므로, 쌍마다 history를 펼쳐 내적한 뒤 쌍 단위로 되합치면 된다.
    def predict_pairs(self, user_ids, game_ids, chunk: int = 20_000) -> np.ndarray:
        targets = np.array([self.game_index[int(g)] for g in game_ids], dtype=np.intp)
        users = np.asarray(user_ids)

        # history table은 user_id로 정렬되어 있으므로, 각 쌍의 user 구간을 searchsorted로 한 번에 찾는다
        hist_users = self.history.table.index.to_numpy()
        hist_rows = self.history.table["row"].to_numpy()
        hist_ratings = self.history.table["rating"].to_numpy()
        lo = np.searchsorted(hist_users, users, side="left")
        counts = np.searchsorted(hist_users, users, side="right") - lo

        # 펼친 배열이 [sum(counts), d]까지 커지므로 쌍을 나눠서 처리 (평가 이력이 없는 user는 0으로 남음)
        preds = np.zeros(len(users), dtype=np.float32)
        for start in range(0, len(users), chunk):
            end = min(start + chunk, len(users))
            n = counts[start:end]
            total = int(n.sum())
            if total == 0:
                continue

            # 각 쌍을 그 user의 history 길이만큼 복제: pair_of[k] = k번째 항목이 속한 쌍의 번호
            pair_of = np.repeat(np.arange(end - start), n)
            within = np.arange(total) - np.repeat(np.cumsum(n) - n, n)   # user 구간 내 offset
            idx = np.repeat(lo[start:end], n) + within

            sims = np.einsum("ij,ij->i", self.V[targets[start:end][pair_of]], self.V[hist_rows[idx]])
            numerator = np.bincount(pair_of, weights=sims * hist_ratings[idx], minlength=end - start)
            denominator = np.bincount(pair_of, weights=sims, minlength=end - start)
            preds[start:end] = numerator / (denominator + EPS)

        return preds

    # 아직 평가하지 않은 game 중 상위 k개를 [(game_id, score), ...]로 반환
    def recommend(self, user_id: int, k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.predict_all(user_id)
        rated_rows, _ = self.history.get(user_id)
        scores[rated_rows] = -np.inf                    # 이미 interact한 game은 추천 대상에서 제외

        # game_ids가 오름차순이므로 stable sort를 쓰면 동점일 때 game_id 오름차순이 유지됨
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(self.game_ids[row]), float(scores[row])) for row in order]


# 4-c: 두 ItemCF의 예측 점수를 alpha로 가중합
class WeightedItemCF:
    def __init__(self, semantic: ItemCF, tfidf: ItemCF, alpha: float = ALPHA):
        self.semantic = semantic
        self.tfidf = tfidf
        self.alpha = alpha

    def predict_all(self, user_id: int) -> np.ndarray:
        return (self.alpha * self.semantic.predict_all(user_id)
                + (1 - self.alpha) * self.tfidf.predict_all(user_id))

    def predict(self, user_id: int, game_id: int) -> float:
        return float(self.predict_all(user_id)[self.semantic.game_index[game_id]])

    def recommend(self, user_id: int, k: int = TOP_K) -> list[tuple[int, float]]:
        scores = self.predict_all(user_id)

        # game_ids가 오름차순이므로 stable sort를 쓰면 동점일 때 game_id 오름차순이 유지됨
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(self.semantic.game_ids[row]), float(scores[row])) for row in order]

    def predict_pairs(self, user_ids, game_ids) -> np.ndarray:
        return (self.alpha * self.semantic.predict_pairs(user_ids, game_ids)
                + (1 - self.alpha) * self.tfidf.predict_pairs(user_ids, game_ids))

    def score_rmse(self, ratings: pd.DataFrame) -> float:
        # ratings에 있는 모든 (user_id, game_id) 쌍에 대해 예측 점수와 실제 점수의 RMSE를 계산
        preds = self.predict_pairs(ratings["user_id"].to_numpy(), ratings["game_id"].to_numpy())
        return float(np.sqrt(np.mean((preds - ratings["rating"].to_numpy(dtype=np.float32)) ** 2)))

# games_metadata.json을 읽어 game_id순으로 정렬된 game_id, description, tags를 반환
# json format: {"game_id":ID, "description":"...", "tags":["A","B", ...]}
def load_game_data()  -> tuple[np.ndarray, list[str], list[list[str]]]:
    metadata = []

    # read single line and parse as json.
    with open(METADATA_DATA, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            metadata.append(json.loads(line))

    # game_id순으로 정렬
    metadata = sorted(metadata, key=lambda x: x["game_id"])
    game_ids = np.array([m["game_id"] for m in metadata])
    descriptions = [m["description"] for m in metadata]
    tags = [m["tags"] for m in metadata]

    return game_ids, descriptions, tags

# ratings.csv를 읽어 user_id, game_id, rating을 담은 DataFrame 반환
def load_ratings_data() -> pd.DataFrame:
    ratings = pd.read_csv(RATINGS_DATA)
    return ratings

# description을 all-MiniLM-L6-v2로 벡터화. shape=[game 수, 384]
# 인코딩에 수 분이 걸리므로 결과를 npy로 캐싱. 캐시가 없거나 game 수가 맞지 않으면 다시 만듦
# (채점 환경에는 캐시가 없으므로 최초 1회 인코딩 후 생성됨)
def build_semantic_matrix(descriptions: list[str]) -> np.ndarray:
    if os.path.exists(SEMANTIC_CACHE):
        cached = np.load(SEMANTIC_CACHE)
        if cached.shape[0] == len(descriptions):
            print(f"[cache] reuse {SEMANTIC_CACHE} (shape={cached.shape})")
            return cached
        print(f"[cache] size mismatch ({cached.shape[0]} != {len(descriptions)}), re-encoding")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = np.array(model.encode(descriptions, show_progress_bar=True))

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(SEMANTIC_CACHE, embeddings)
    return embeddings

def build_tfidf_matrix(tags: list[list[str]]) -> np.ndarray:
    unique_tags = sorted({tag for game_tags in tags for tag in game_tags})
    tag_col = {tag: i for i, tag in enumerate(unique_tags)}

    # IDF = log10(total_count / tag_count)
    #   total_count: 전체 game 개수, tag_count: 해당 tag가 등록된 game 개수
    total_count = len(tags)
    tag_counts = Counter()

    for game_tags in tags:
        tag_counts.update(set(game_tags))
    idf = {tag: np.log10(total_count / tag_counts[tag]) for tag in unique_tags}

    # TF = n(d, t) / n(d)
    # 각 game이 실제로 가진 tag만 순회하고, 해당 열에 TF-IDF를 누적
    tfidf_matrix = np.zeros((len(tags), len(unique_tags)))
    for row, game_tags in enumerate(tags):
        n_d = len(game_tags)
        if n_d == 0:
            continue
        for tag in game_tags:
            tfidf_matrix[row, tag_col[tag]] += idf[tag] / n_d

    return tfidf_matrix

# read input_recommendation.txt : 한 줄에 user_id 하나씩
def read_recommendation_input() -> list[int]:
    with open(RECOMMEND_INPUT, "r") as f:
        return [int(l.strip()) for l in f.readlines() if l.strip()]


# read input_score_prediction.txt : 한 줄에 "user_id;game_id"
def read_prediction_input() -> list[tuple[int, int]]:
    pairs = []
    with open(PREDICTION_INPUT, "r") as f:
        for l in f.readlines():
            l = l.strip()
            if not l:
                continue
            uid, gid = l.split(";")
            pairs.append((int(uid), int(gid)))
    return pairs


# lines를 그대로 파일에 씀 (한 줄에 하나씩, 이미 "user_id,game_id,score" 형태로 포맷된 문자열)
def write_output(lines: list[str], filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for line in lines:
            f.write(line + "\n")

#### TODO: 아래 함수들을 실제 구현으로 교체하세요 ####

# 필드(user_id, game_id, prediction_score)  간  구분자는  공백  없이  콤마( , )를  사용

def semantics(user_ids: list[int], recommendor: ItemCF) -> list[str]:
    result = []
    for i in user_ids:
        predictions = recommendor.recommend(i, TOP_K)
        for game_id, score in predictions:
            result.append("{},{},{:.4f}".format(i, game_id, score))
    return result

def tfidf(user_ids: list[int], recommendor: ItemCF) -> list[str]:
    result = []
    for i in user_ids:
        predictions = recommendor.recommend(i, TOP_K)
        for game_id, score in predictions:
            result.append("{},{},{:.4f}".format(i, game_id, score))
    return result

def weighted(pairs: list[tuple[int, int]], recommendor: WeightedItemCF) -> list[str]:
    result = []
    for user_id, game_id in pairs:
        score = recommendor.predict(user_id, game_id)
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

#### TODO end ####