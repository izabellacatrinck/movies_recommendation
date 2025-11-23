from __future__ import annotations

import os
import random  

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from pandas import DataFrame

load_dotenv()

TMDB_BASE_URL = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_POSTER = "https://via.placeholder.com/342x513?text=No+Image"


def _build_poster_url(poster_path) -> str:
    """
    Aceita:
      - path TMDB (ex.: '/hldXwwViSfHJS0kIJr07KBGmHJI.jpg' ou 'hldXww...')
      - URL completa (http/https)
      - NaN / vazio -> placeholder
    """
    if not isinstance(poster_path, str):
        return PLACEHOLDER_POSTER

    poster_path = poster_path.strip()
    if not poster_path:
        return PLACEHOLDER_POSTER
    if poster_path.startswith("http://") or poster_path.startswith("https://"):
        return poster_path

    # path TMDB
    if not poster_path.startswith("/"):
        poster_path = "/" + poster_path

    return TMDB_BASE_URL + poster_path


def _get_backend_url() -> str | None:
    """
    BACKEND_URL deve ser algo como: http://127.0.0.1:8000
    (sem /api no final; aqui a gente acrescenta /api/...).
    """
    backend_url = os.getenv("BACKEND_URL") or st.secrets.get("backend", {}).get("url")
    if not backend_url:
        return None
    return backend_url.rstrip("/")


def _render_similar_carousel(
    user_id: int,
    ratings_df: DataFrame,
    movies_df: DataFrame,
) -> None:
    """
    Mostra um carrossel com filmes similares a um dos filmes avaliados
    pelo usuário, usando a rota /api/{movie_id}/similar.
    """
    backend_url = _get_backend_url()
    if not backend_url:
        st.caption("BACKEND_URL não configurado; não foi possível mostrar filmes parecidos.")
        return

    hist_raw = ratings_df[ratings_df["userId"].astype(str) == str(user_id)].copy()
    if hist_raw.empty:
        return

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
        return

    hist = hist_valid.merge(
        movies_df,
        left_on="movieId",
        right_on="id",
        how="left",
    )

    try:
        base_row = hist.sample(1, random_state=random.randint(0, 10_000)).iloc[0]
    except ValueError:
        return

    try:
        base_id = int(base_row["movieId"])
    except (TypeError, ValueError):
        return

    base_title = str(base_row.get("title") or "este filme")
    st.markdown(f"### 🎞️ Filmes parecidos com **{base_title}**")

    url = f"{backend_url}/api/{base_id}/similar"

    try:
        resp = requests.get(url, params={"k": 10}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        st.caption(
            f"Não foi possível carregar filmes parecidos agora "
            f"(id={base_id}, status={getattr(e.response, 'status_code', '?')})."
        )
        return
    except requests.RequestException:
        st.caption("Não foi possível carregar filmes parecidos agora.")
        return

    if not data:
        st.caption("Nenhum filme parecido encontrado.")
        return

    sim_df = pd.DataFrame(data)
    # se vier como movie_id, renomeia pra id
    if "movie_id" in sim_df.columns and "id" not in sim_df.columns:
        sim_df = sim_df.rename(columns={"movie_id": "id"})

    # se por algum motivo não tiver id, mostra só texto simples
    if "id" not in sim_df.columns:
        cols = st.columns(len(sim_df[:3]))
        for i, (_, row) in enumerate(sim_df[:3].iterrows()):
            with cols[i]:
                st.markdown(f"**{row.get('title', 'Sem título')}**")
                genres = row.get("genres")
                if pd.notna(genres):
                    st.caption(f"🎭 {genres}")
        st.divider()
        return

    # junta com movies_df pra pegar poster_path, release_date etc.
    view = sim_df.merge(
        movies_df,
        on="id",
        how="left",
        suffixes=("", "_cat"),
    )

    cols = st.columns(3)
    for i, (_, row) in enumerate(view.head(3).iterrows()):
        with cols[i]:
            st.image(
                _build_poster_url(row.get("poster_path")),
                use_container_width=True,
            )
            st.markdown(f"**{row.get('title', 'Sem título')}**")
            if pd.notna(row.get("genres")):
                st.caption(f"🎭 {row['genres']}")
            rel = row.get("release_date")
            if pd.notna(rel):
                year = str(rel)[:4]
                st.caption(f"Ano: {year}")

    st.divider()


def render_recommendation_card(row: pd.Series) -> None:
    """Renderiza um card de filme recomendado."""
    with st.container():
        poster_url = _build_poster_url(row.get("poster_path"))
        st.image(poster_url, width="stretch")

        title = row.get("title") or "Sem título"
        st.markdown(f"**{title}**")

        genres = row.get("genres")
        if pd.notna(genres):
            st.caption(str(genres))

        col1, col2 = st.columns(2)
        with col1:
            release_date = row.get("release_date")
            year_str = "-"
            if pd.notna(release_date):
                try:
                    year_str = str(release_date)[:4]
                except Exception:
                    year_str = str(release_date)
            st.caption(f"Ano: {year_str}")

        with col2:
            movie_id = row.get("movie_id", None)
            if pd.isna(movie_id):
                movie_id = row.get("id", None)

            if pd.notna(movie_id):
                try:
                    st.caption(f"ID: {int(movie_id)}")
                except (TypeError, ValueError):
                    st.caption(f"ID: {movie_id}")


def get_recommendations(
    user_id: str | int,
    top_n: int,
) -> tuple[DataFrame | None, dict, dict]:
    """
    Chama o backend em /api/recommendations e retorna:
    - DataFrame de recomendações
    - dict de métricas
    - dict com infos extras (likes, flags, etc.)
    """
    try:
        backend_url = _get_backend_url()
        if not backend_url:
            st.error("BACKEND_URL não configurada (.env ou st.secrets).")
            return None, {}, {}

        url = f"{backend_url}/api/recommendations"

        params = {
            "user_id": int(user_id),
            "k": int(top_n),
        }

        r: requests.Response = requests.get(
            url,
            params=params,
            timeout=120,
        )

        # caso especial: backend retornando 400 para "poucos ratings"
        if r.status_code == 400:
            msg = "Avalie mais filmes para receber recomendações."
            try:
                data_err = r.json()
                detail = data_err.get("detail") or data_err.get("message")
                if isinstance(detail, str) and detail.strip():
                    msg = detail
            except Exception:
                pass
            return None, {}, {
                "needs_more_ratings": True,
                "message": msg,
            }

        r.raise_for_status()
        data = r.json()

        recs = data.get("recommendations", []) or []
        metrics = data.get("metrics", {}) or {}
        info = {
            "total_likes": data.get("total_likes"),
            "train_likes": data.get("train_likes"),
            "test_likes": data.get("test_likes"),
        }

        if not recs:
            return None, metrics, info

        rec_df = pd.DataFrame(recs)
        return rec_df, metrics, info

    except requests.RequestException as e:
        st.error(f"Erro ao buscar recomendações: {e}")
        return None, {}, {}
    except Exception as e:
        st.error(f"Erro inesperado ao processar recomendações: {e}")
        return None, {}, {}


def render(user_df: DataFrame, movies_df: DataFrame) -> None:
    """
    Renderiza a página de recomendações.

    Parâmetros:
    - user_df: ratings_final_df (colunas: userId, movieId, rating)
    - movies_df: movies_final_df (colunas: id, title, genres, poster_path, release_date, ...)
    """
    st.subheader("✨ Recomendações")
    st.caption(
        "Sistema de recomendação **híbrido** (conteúdo + filtragem colaborativa) "
        "com avaliação por usuário (precision / recall / F1)."
    )

    user_id = str(st.session_state.get("current_user", ""))

    available_users = (
        user_df["userId"]
        .dropna()
        .astype(int)
        .sort_values()
        .astype(str)
        .unique()
    )

    if not user_id:
        if len(available_users) == 0:
            st.warning("Nenhum usuário disponível em ratings_final_df.")
            return
        user_id = str(
            st.selectbox(
                "Selecione um usuário:",
                options=available_users,
                index=0,
            )
        )
    else:
        if user_id not in available_users and len(available_users) > 0:
            user_id = available_users[0]

    col1, _ = st.columns(2)
    with col1:
        top_n: int = st.slider("Número de recomendações (k)", 1, 20, 10)

    if st.button("🔍 Ver recomendações", use_container_width=True):
        if not user_id:
            st.warning("Selecione um usuário válido.")
            return

        with st.spinner("Buscando recomendações..."):
            rec_df, metrics, info = get_recommendations(user_id, top_n)

            if info.get("needs_more_ratings"):
                st.warning(info.get("message", "Avalie mais filmes para receber recomendações."))
                return

            if rec_df is None or rec_df.empty:
                st.warning("Sem recomendações para este usuário.")
                return

            catalog: DataFrame = movies_df.copy()
            view: DataFrame = (
                rec_df.merge(
                    catalog,
                    left_on="movie_id",
                    right_on="id",
                    how="left",
                    suffixes=("", "_cat"),
                )
                .drop_duplicates(subset=["movie_id"])
            )

            if "title_cat" in view.columns:
                if "title" not in view.columns:
                    view["title"] = view["title_cat"]
                else:
                    view["title"] = view["title"].fillna(view["title_cat"])

            if "genres_cat" in view.columns:
                if "genres" not in view.columns:
                    view["genres"] = view["genres_cat"]
                else:
                    view["genres"] = view["genres"].fillna(view["genres_cat"])
        try:
            uid_int = int(user_id)
        except ValueError:
            uid_int = None

        if uid_int is not None:
            _render_similar_carousel(uid_int, user_df, movies_df)

        # ====== Métricas ======
        st.markdown("#### Métricas de avaliação (train/test split do usuário)")

        m_precision = metrics.get("precision_at_k")
        m_recall = metrics.get("recall_at_k")
        m_f1 = metrics.get("f1_at_k")

        def fmt_pct(v):
            if isinstance(v, (int, float)):
                return f"{v * 100:.1f}%"
            return "-"

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Precision@k", fmt_pct(m_precision))
        with col_m2:
            st.metric("Recall@k", fmt_pct(m_recall))
        with col_m3:
            st.metric("F1@k", fmt_pct(m_f1))

        total_likes = info.get("total_likes")
        train_likes = info.get("train_likes")
        test_likes = info.get("test_likes")

        st.caption(
            f"Likes do usuário usados na avaliação: "
            f"total={total_likes}, train={train_likes}, test={test_likes}"
        )

        # ====== Lista de filmes recomendados ======
        st.markdown("#### Filmes recomendados")
        cols = st.columns(3)
        for i, (_, row) in enumerate(view.iterrows()):
            with cols[i % 3]:
                render_recommendation_card(row)

        st.success(f"✨ {len(view)} recomendações exibidas para usuário {user_id}")
