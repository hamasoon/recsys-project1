# ALPHA 검증: 10% holdout을 떼고 나머지 90%를 interaction 단위 5-fold로 나눠 alpha를 고른 뒤, holdout에서 0.5와 비교
# 실행: python validate.py (채점용 main.py와 별개. main.py의 cache/ 재사용)
import sys
import numpy as np
import pandas as pd
import sentence_transformers
import main

SEED = 0
HOLDOUT_RATIO = 0.1
N_FOLDS = 5
N_BOOTSTRAP = 1000
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)      # 0 < alpha < 1
I_HALF = int(np.flatnonzero(GRID == 0.5)[0])


# train 평점만으로 이력을 만들고, target 쌍에 대한 alpha별 쌍 단위 제곱오차 [len(GRID), n]와 warm 여부를 반환
def squared_errors(train: pd.DataFrame, target: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    history = main.UserHistory(train, game_index)
    model = main.WeightedItemCF(main.ItemCF(semantic_matrix, game_ids, game_index, history),
                                main.ItemCF(tfidf_matrix, game_ids, game_index, history))
    users, games = target["user_id"].to_numpy(), target["game_id"].to_numpy()
    y = target["rating"].to_numpy(dtype=np.float64)
    sem = model.semantic.predict_pairs(users, games)     # 두 모델 점수는 한 번만 계산하고 alpha만 바꿔 재사용
    tfidf = model.tfidf.predict_pairs(users, games)

    errors = np.empty((len(GRID), len(y)))
    for i, alpha in enumerate(GRID):
        model.alpha = alpha
        errors[i] = (model.combine(users, sem, tfidf) - y) ** 2
    return errors, history.contains(users)


def rmse(sq_err: np.ndarray) -> np.ndarray:
    return np.sqrt(sq_err.mean(axis=-1))


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
          f"sentence-transformers {sentence_transformers.__version__}")
    game_ids, descriptions, tags = main.load_game_data()
    ratings = main.load_ratings_data()
    game_index = main.build_game_index(game_ids)
    semantic_matrix = main.build_semantic_matrix(descriptions)
    tfidf_matrix = main.build_tfidf_matrix(tags)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ratings))
    n_holdout = int(len(ratings) * HOLDOUT_RATIO)
    holdout_idx, dev_idx = perm[:n_holdout], perm[n_holdout:]
    folds = np.array_split(dev_idx, N_FOLDS)
    print(f"ratings {len(ratings)}: holdout {len(holdout_idx)}, dev {len(dev_idx)} ({N_FOLDS}-fold)")

    # 5-fold: fold별 제곱오차 합을 누적해 전체 제곱오차가 최소인 공통 alpha 선택
    sse, sse_warm, n, n_warm = np.zeros(len(GRID)), np.zeros(len(GRID)), 0, 0
    for k in range(N_FOLDS):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        errors, warm = squared_errors(ratings.iloc[train_idx], ratings.iloc[folds[k]])
        sse += errors.sum(axis=1); sse_warm += errors[:, warm].sum(axis=1)
        n += len(warm); n_warm += int(warm.sum())
        r = rmse(errors); best = int(np.argmin(r))
        print(f"fold {k}: val {len(warm)}, warm {warm.mean():.3f}, "
              f"alpha=0.5 {r[I_HALF]:.4f}, best alpha={GRID[best]} {r[best]:.4f}")
    r_all, r_warm = np.sqrt(sse / n), np.sqrt(sse_warm / n_warm)
    best = int(np.argmin(r_all))
    alpha = GRID[best]
    print(f"CV pooled: alpha=0.5 {r_all[I_HALF]:.4f} (warm {r_warm[I_HALF]:.4f}), "
          f"best alpha={alpha} {r_all[best]:.4f} (warm {r_warm[best]:.4f})")

    # holdout: dev 90% 전체를 이력으로 쓰고, 고른 alpha와 0.5 비교. 쌍 단위 paired bootstrap으로 차이 확인
    errors, warm = squared_errors(ratings.iloc[dev_idx], ratings.iloc[holdout_idx])
    r_all, r_warm = rmse(errors), rmse(errors[:, warm])
    diff = errors[best] - errors[I_HALF]
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(N_BOOTSTRAP)])
    print(f"holdout: n {len(diff)}, warm {warm.mean():.3f}, "
          f"alpha=0.5 {r_all[I_HALF]:.4f} (warm {r_warm[I_HALF]:.4f}), "
          f"alpha={alpha} {r_all[best]:.4f} (warm {r_warm[best]:.4f}), "
          f"holdout-best alpha={GRID[int(np.argmin(r_all))]}")
    print(f"holdout MSE diff (alpha={alpha} - 0.5): {diff.mean():.5f}, "
          f"bootstrap 95% CI [{np.percentile(boot, 2.5):.5f}, {np.percentile(boot, 97.5):.5f}], "
          f"P(diff < 0) {np.mean(boot < 0):.3f}")
