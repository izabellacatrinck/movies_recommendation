# backend/api/routes_ratings.py
from __future__ import annotations

import os
import pandas as pd

from fastapi import APIRouter, Depends, Request, HTTPException

from ..content_recommender import ContentKNNRecommender
from ..services.schemas import RatingCreate  # schema com user_id, movie_id, rating

router = APIRouter(tags=["ratings"])

# mesmo arquivo usado pelo modelo
RATINGS_RUNTIME_PATH = "data/ratings_final_df.csv"


def get_recommender(request: Request) -> ContentKNNRecommender:
    recommender = getattr(request.app.state, "recommender", None)
    if recommender is None:
        raise HTTPException(status_code=500, detail="Recommender não inicializado")
    return recommender


@router.post("/ratings")
def create_rating(
    payload: RatingCreate,
    recommender: ContentKNNRecommender = Depends(get_recommender),
):
    """
    Registra uma avaliação e salva em CSV + atualiza o dataframe em memória.

    Regras:
    - Remove linhas dummy do usuário (movieId <= 0 ou rating <= 0).
    - Remove qualquer avaliação anterior do mesmo (userId, movieId).
    - Adiciona a nova avaliação.
    - Atualiza recommender.df_ratings em memória no formato interno.
    """
    user_id = int(payload.user_id)
    movie_id = int(payload.movie_id)
    rating = float(payload.rating)

    # 1) Carrega o CSV completo (ou cria vazio)
    if os.path.exists(RATINGS_RUNTIME_PATH):
        df = pd.read_csv(RATINGS_RUNTIME_PATH)
    else:
        df = pd.DataFrame(columns=["userId", "movieId", "rating"])

    # garante colunas necessárias
    for col in ["userId", "movieId", "rating"]:
        if col not in df.columns:
            df[col] = pd.Series(dtype=float if col == "rating" else int)

    if not df.empty:
        # normaliza tipos p/ evitar NaN estranhos
        df["userId"] = pd.to_numeric(df["userId"], errors="coerce").fillna(0).astype(int)
        df["movieId"] = pd.to_numeric(df["movieId"], errors="coerce").fillna(0).astype(int)
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)

    # 2) Remove dummies e rating antigo desse user/filme
    mask_user = df["userId"] == user_id

    # linhas dummy do usuário (movieId <= 0 ou rating <= 0)
    mask_dummy_user = mask_user & ((df["movieId"] <= 0) | (df["rating"] <= 0))

    # linhas antigas desse mesmo filme p/ esse user
    mask_same_movie = mask_user & (df["movieId"] == movie_id)

    # mantemos tudo que NÃO é dummy e NÃO é o rating antigo do mesmo filme
    df_clean = df[~(mask_dummy_user | mask_same_movie)].copy()

    # 3) Adiciona a nova avaliação
    new_row_csv = pd.DataFrame(
        [{"userId": user_id, "movieId": movie_id, "rating": rating}]
    )
    df_final = pd.concat([df_clean, new_row_csv], ignore_index=True)

    # 4) Salva o CSV inteiro de volta (sem header duplicado)
    df_final.to_csv(RATINGS_RUNTIME_PATH, index=False)

    # 5) Atualiza df_ratings em memória no formato interno (user_id/movie_id/rating)
    df_internal = df_final.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
        }
    )

    df_internal["user_id"] = pd.to_numeric(df_internal["user_id"], errors="coerce").fillna(0).astype(int)
    df_internal["movie_id"] = pd.to_numeric(df_internal["movie_id"], errors="coerce").fillna(0).astype(int)
    df_internal["rating"] = pd.to_numeric(df_internal["rating"], errors="coerce").fillna(0.0)

    # aqui NÃO fazemos o filtro de "common" com movies, isso fica a cargo do fit()
    recommender.df_ratings = df_internal[["user_id", "movie_id", "rating"]]

    return {
        "status": "ok",
        "user_id": user_id,
        "movie_id": movie_id,
        "rating": rating,
    }
