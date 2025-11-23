from __future__ import annotations

import pandas as pd
import streamlit as st
from pandas import DataFrame

from utils import paginate
from .catalog import _build_poster_url 


def render_movie_card(row: pd.Series) -> None:
    """Card de filmes avaliados."""
    with st.container():
        st.image(_build_poster_url(row.get("poster_path")), use_container_width=True)

        st.markdown(f"**{row.get('title', 'Sem título')}**")

        if pd.notna(row.get("genres")):
            st.caption(f"🎭 {row['genres']}")

        col1, col2 = st.columns(2)
        with col1:
            rating = row.get("rating")
            if pd.notna(rating):
                st.metric("Nota", f"{float(rating):.1f}")
        with col2:
            rel = row.get("release_date")
            if pd.notna(rel):
                year = str(rel)[:4]
                st.caption(f"Ano: {year}")
            if pd.notna(row.get("id")):
                st.caption(f"ID: {int(row['id'])}")


def render(ratings_df: DataFrame, movies_df: DataFrame) -> None:
    """Página 'Minhas avaliações'."""
    st.subheader("⭐ Minhas avaliações")

    uid = st.session_state.get("current_user")
    if not uid:
        st.info("Selecione um usuário na barra lateral.")
        return

    required = {"userId", "movieId", "rating"}
    if not required.issubset(ratings_df.columns):
        st.warning(
            "ratings_final_df.csv não tem as colunas esperadas (userId, movieId, rating)."
        )
        return

    # histórico bruto do usuário
    hist_raw = ratings_df[ratings_df["userId"].astype(str) == str(uid)].copy()
    if hist_raw.empty:
        st.caption("Você ainda não tem avaliações registradas.")
        return

    # limpa placeholders / linhas inválidas
    hist_valid = hist_raw.copy()
    hist_valid = hist_valid[
        hist_valid["movieId"].notna()
        & hist_valid["rating"].notna()
    ]

    hist_valid["movieId"] = pd.to_numeric(hist_valid["movieId"], errors="coerce")
    hist_valid["rating"] = pd.to_numeric(hist_valid["rating"], errors="coerce")

    hist_valid = hist_valid[
        (hist_valid["movieId"] > 0)
        & (hist_valid["rating"] > 0)
    ]

    if hist_valid.empty:
        st.caption(
            "Você ainda não tem avaliações reais registradas. "
            "Avalie alguns filmes no catálogo."
        )
        return

    # junta com filmes
    hist = hist_valid.merge(
        movies_df,
        left_on="movieId",
        right_on="id",
        how="left",
    )

    hist = hist.sort_values(["rating", "title"], ascending=[False, True])

    st.markdown("### Seus filmes avaliados")

    page: DataFrame = paginate(hist, page_size=6, key="my_ratings_page")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(page.iterrows()):
        with cols[idx % 3]:
            render_movie_card(row)

    st.divider()
