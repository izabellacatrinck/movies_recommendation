from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from ..content_recommender import ContentKNNRecommender
import pandas as pd

router = APIRouter(tags=["movies"])


def get_recommender(request: Request) -> ContentKNNRecommender:
    return request.app.state.recommender


@router.get("/", summary="Listar filmes")
def list_movies(
    limit: int = 50,
    offset: int = 0,
    recommender: ContentKNNRecommender = Depends(get_recommender),
):
    df = recommender.df_movies.iloc[offset: offset + limit]
    return df[["id", "title", "genres"]].to_dict(orient="records")


@router.get("/{movie_id}", summary="Detalhar um filme")
def get_movie(
    movie_id: int,
    recommender: ContentKNNRecommender = Depends(get_recommender),
):
    df = recommender.df_movies
    row = df[df["id"] == movie_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    rec = row.iloc[0]
    return {
        "id": int(rec["id"]),
        "title": rec["title"],
        "genres": rec.get("genres"),
        "overview": rec.get("overview") or rec.get("content"),
        "companies_text": rec.get("companies_text"),
        "popularity": float(rec["popularity"]) if pd.notna(rec["popularity"]) else None,
    }


@router.get("/{movie_id}/similar", summary="Filmes similares (item-item)")
def get_similar_movies(
    movie_id: int,
    k: int = 10,
    recommender: ContentKNNRecommender = Depends(get_recommender),
):
    try:
        df_sim = recommender.similar_movies(movie_id, top_n=k)
    except ValueError:
        raise HTTPException(status_code=404, detail="Movie not found")

    return df_sim.to_dict(orient="records")
