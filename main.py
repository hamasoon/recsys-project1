# Project 1 : Content-based Recommendation
# 2026-00000 홍길동
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

DATA_DIR = "data"
RECOMMEND_INPUT = "input_recommendation.txt"       # 4-b: user_id 목록
PREDICTION_INPUT = "input_score_prediction.txt"    # 4-c: user_id;game_id 목록
OUTPUT_DIR = "results"
TOP_K = 20


# read input_recommendation.txt : 한 줄에 user_id 하나씩
def read_recommendation_input():
    with open(RECOMMEND_INPUT, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


# read input_score_prediction.txt : 한 줄에 "user_id;game_id"
def read_prediction_input():
    pairs = []
    with open(PREDICTION_INPUT, "r") as f:
        for l in f.readlines():
            l = l.strip()
            if not l:
                continue
            uid, gid = l.split(";")
            pairs.append((uid, gid))
    return pairs


# lines를 그대로 파일에 씀 (한 줄에 하나씩, 이미 "user_id,game_id,score" 형태로 포맷된 문자열)
def write_output(lines, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, filename), "w") as f:
        for line in lines:
            f.write(line + "\n")

#### TODO: 아래 함수들을 실제 구현으로 교체하세요 ####

def semantics(ids):
    # test implementation — user당 TOP_K개, "user_id,game_id,prediction_score" 형태
    prediction = []
    for i in ids:
        prediction += ["{},{},{:.4f}".format(i, g, 3.5) for g in range(1, TOP_K + 1)]
    return prediction


def tfidf(ids):
    # test implementation
    prediction = []
    for i in ids:
        prediction += ["{},{},{:.4f}".format(i, g, 3.5) for g in range(1, TOP_K + 1)]
    return prediction


def weighted(pairs):
    # test implementation — 4-c: TF-IDF/Semantics 예측을 가중결합해서
    # input_score_prediction.txt로 받은 (user_id, game_id) 쌍 각각에 대한 prediction_score만 계산.
    # top-K로 추리는 게 아니라 주어진 쌍 개수만큼 한 줄씩 출력.
    return ["{},{},{:.4f}".format(uid, gid, 3.5) for uid, gid in pairs]

user_ids = read_recommendation_input()
result_sem = semantics(user_ids)
result_tfidf = tfidf(user_ids)

pairs = read_prediction_input()
result_weighted = weighted(pairs)

write_output(result_sem, "semantics_output.txt")
write_output(result_tfidf, "tfidf_output.txt")
write_output(result_weighted, "score_prediction_output.txt")
#### TODO end ####