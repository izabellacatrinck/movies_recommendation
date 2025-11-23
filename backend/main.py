# backend/main.py
from __future__ import annotations

import os
import pandas as pd

from fastapi import FastAPI

from .content_recommender import ContentKNNRecommender, ContentKNNConfig
from .api.routes_rec import router as recs_router
from .api.routes_users import router as users_router
from .api.routes_ratings import router as ratings_router
from .api.routes_movies import router as movies_router

RATINGS_RUNTIME_PATH = "data/ratings_runtime.csv"

app = FastAPI(title="Movie Recommender API")


@app.on_event("startup")
def startup_event():
    cfg = ContentKNNConfig(
        movies_csv_path="data/movies_final_df.csv",
        ratings_csv_path="data/ratings_final_df.csv",
    )

    recommender = ContentKNNRecommender(cfg)
    recommender.fit()

    # Carregar ratings extras (runtime) se existir
    if os.path.exists(RATINGS_RUNTIME_PATH):
        extra = pd.read_csv(RATINGS_RUNTIME_PATH)

        # garantir nomes no padrão original, depois renomear pro interno
        # (userId, movieId, rating) -> (user_id, movie_id, rating)
        if not extra.empty:
            extra = extra.rename(
                columns={
                    "userId": "user_id",
                    "movieId": "movie_id",
                }
            )
            extra["user_id"] = extra["user_id"].astype(int)
            extra["movie_id"] = extra["movie_id"].astype(int)

            recommender.df_ratings = pd.concat(
                [recommender.df_ratings, extra[["user_id", "movie_id", "rating"]]],
                ignore_index=True,
            )

    app.state.recommender = recommender


# rotas
app.include_router(recs_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(ratings_router, prefix="/api")
app.include_router(movies_router, prefix="/api")