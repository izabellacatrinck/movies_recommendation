from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd

from .content_recommender import ContentKNNRecommender, ContentKNNConfig

MIN_RATING_LIKE = 4.0   # nota mínima pra considerar "gostou" no TEST
MIN_LIKES_USER  = 1    # mínimo de likes pra usuário entrar na avaliação
TEST_SIZE       = 0.4   # % das curtidas que vão pro teste
K_RECS          = 10    


def precision_recall_f1_at_k(
    recommended_ids: List[int],
    relevant_ids: List[int],
    k: int,
) -> Dict[str, float]:
    """
    Calcula Precision@K, Recall@K e F1@K.
    """
    recommended_ids = recommended_ids[:k]
    recommended_set = set(recommended_ids)
    relevant_set = set(relevant_ids)

    hits = len(recommended_set & relevant_set)

    precision = hits / float(k) if k > 0 else 0.0
    recall = hits / float(len(relevant_set)) if len(relevant_set) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def split_genres(s):
    if pd.isna(s) or not isinstance(s, str):
        return []
    return s.lower().split()


def evaluate_user_recommendations(
    recommender: ContentKNNRecommender,
    user_id: int,
    k: int = K_RECS,
    min_rating_like: float = MIN_RATING_LIKE,
    test_size: float = TEST_SIZE,
) -> Dict[str, Any]:
    """
    Avalia o recomendador para UM usuário específico
    e devolve recomendação + métricas em formato pronto pra API.
    """
    df_movies = recommender.df_movies
    df_ratings_full = recommender.df_ratings

    # --- ratings desse usuário (no dataset completo) ---
    user_ratings_all = df_ratings_full[df_ratings_full["user_id"] == user_id]
    user_likes_all = user_ratings_all[user_ratings_all["rating"] >= min_rating_like].copy()

    total_likes = user_likes_all.shape[0]
    if total_likes < 2:
        return {
            "user_id": user_id,
            "ok": False,
            "message": "Usuário com likes insuficientes para avaliação.",
        }

    # --- split train/test nas curtidas ---
    user_likes_all = user_likes_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_test = max(1, int(len(user_likes_all) * test_size))

    df_test = user_likes_all.iloc[:n_test].copy()
    df_train = user_likes_all.iloc[n_test:].copy()

    # gabarito (filmes relevantes de TEST)
    df_test_movies = df_test.merge(
        df_movies,
        left_on="movie_id",
        right_on=recommender.config.movie_id_column,
        how="left",
    )[["movie_id", "title", "rating"]]

    # --- ajusta temporariamente df_ratings para NÃO usar os filmes de teste ---
    df_ratings_orig = recommender.df_ratings.copy()

    test_movie_ids = set(df_test["movie_id"].astype(int).tolist())
    mask_user = df_ratings_orig["user_id"] == user_id
    mask_test_movies = df_ratings_orig["movie_id"].isin(test_movie_ids)

    df_ratings_train_view = df_ratings_orig[~(mask_user & mask_test_movies)].reset_index(drop=True)
    recommender.df_ratings = df_ratings_train_view

    # --- recomendações ---
    recs = recommender.recommend_for_user(
        user_id=user_id,
        top_n=k,
        min_rating=min_rating_like,
    )

    # restaura df_ratings original
    recommender.df_ratings = df_ratings_orig

    if recs.empty:
        return {
            "user_id": user_id,
            "ok": False,
            "message": "Nenhuma recomendação retornada para esse usuário.",
        }

    # --- métricas ---
    recommended_ids = recs["movie_id"].astype(int).tolist()
    relevant_ids = df_test["movie_id"].astype(int).tolist()

    metrics = precision_recall_f1_at_k(recommended_ids, relevant_ids, k)
    recs["is_relevant"] = recs["movie_id"].isin(set(relevant_ids))
    df_train_movies = df_train.merge(
        df_movies,
        left_on="movie_id",
        right_on=recommender.config.movie_id_column,
        how="left",
    )
    fav_genres_set = set(
        g for genres in df_train_movies["genres"].apply(split_genres).tolist() for g in genres
    )

    def jaccard_with_fav(genres_str):
        genres_set = set(split_genres(genres_str))
        if not genres_set or not fav_genres_set:
            return 0.0
        inter = len(genres_set & fav_genres_set)
        union = len(genres_set | fav_genres_set)
        return inter / union

    recs["genre_jaccard"] = recs["genres"].apply(jaccard_with_fav)

    # prepara saída em formato "API friendly"
    recommendations_payload = recs[
        ["movie_id", "title", "genres", "score", "genre_jaccard", "is_relevant"]
    ].reset_index(drop=True).to_dict(orient="records")

    relevant_payload = df_test_movies.reset_index(drop=True).to_dict(orient="records")

    return {
        "ok": True,
        "user_id": user_id,
        "k": k,
        "min_rating_like": float(min_rating_like),
        "total_likes": int(total_likes),
        "train_likes": int(df_train.shape[0]),
        "test_likes": int(df_test.shape[0]),
        "metrics": {
            "precision_at_k": metrics["precision"],
            "recall_at_k": metrics["recall"],
            "f1_at_k": metrics["f1"],
        },
        "relevant_items_test": relevant_payload,
        "recommendations": recommendations_payload,
    }
