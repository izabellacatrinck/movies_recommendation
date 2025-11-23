from __future__ import annotations

import os
import pandas as pd

from fastapi import APIRouter, HTTPException

from ..services.schemas import UserCreate 

router = APIRouter(
    prefix="/users",
    tags=["users"],
)

RATINGS_CSV_PATH = "data/ratings_final_df.csv"


def load_ratings_df() -> pd.DataFrame:
    """Carrega o ratings_final_df.csv.

    Se não existir (caso bem raro em dev), cria um DF vazio com as colunas básicas.
    """
    if os.path.exists(RATINGS_CSV_PATH):
        return pd.read_csv(RATINGS_CSV_PATH)

    return pd.DataFrame(columns=["userId", "movieId", "rating"])


@router.post("", status_code=201)
def create_user(payload: UserCreate):
    """
    Cadastra um novo usuário dentro do *mesmo* CSV ratings_final_df.csv.

    - Garante unicidade de userId.
    - Cria UMA linha dummy com userId, movieId=0, rating=0.0.
      (O modelo ignora esse filme 0, pois não existe em movies_final_df.)
      Na primeira avaliação real, essa linha será removida na rota /ratings.
    """
    user_id = int(payload.id)

    df = load_ratings_df()

    if "userId" in df.columns and not df.empty:
        existing_ids = set(df["userId"].dropna().astype(int).unique())
    else:
        existing_ids = set()

    if user_id in existing_ids:
        raise HTTPException(
            status_code=400,
            detail=f"User id {user_id} já existe.",
        )

    if df.empty:
        row_dict = {"userId": user_id, "movieId": 0, "rating": 0.0}
        new_df = pd.DataFrame([row_dict])
        new_df.to_csv(RATINGS_CSV_PATH, index=False)
    else:
        row_dict = {col: None for col in df.columns}
        row_dict["userId"] = user_id
        row_dict["movieId"] = 0
        row_dict["rating"] = 0.0

        new_df = pd.DataFrame([row_dict], columns=df.columns)

        new_df.to_csv(RATINGS_CSV_PATH, mode="a", header=False, index=False)

    return {"status": "ok", "userId": user_id}


@router.get("")
def list_users():
    """
    Retorna TODOS os usuários (userId) encontrados no ratings_final_df.csv.
    """
    df = load_ratings_df()

    if "userId" not in df.columns or df.empty:
        return []

    user_ids = (
        df["userId"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return user_ids
