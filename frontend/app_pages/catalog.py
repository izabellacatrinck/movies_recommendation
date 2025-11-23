from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from pandas import DataFrame
from utils import paginate 

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

    if not poster_path.startswith("/"):
        poster_path = "/" + poster_path

    return TMDB_BASE_URL + poster_path


# -------- modal de avaliação --------
@st.dialog("Avaliar filme")
def show_rating_dialog(movie_id: int, movie_title: str):
    """Exibe modal para usuário avaliar um filme."""
    user_id = st.session_state.get("current_user")
    if not user_id:
        st.warning("⚠️ Selecione um usuário na barra lateral para poder avaliar.")
        if st.button("Fechar"):
            st.rerun()
        return

    st.markdown(f"**{movie_title}**")
    st.markdown("---")

    rating: float = st.slider(
        "Sua nota",
        min_value=0.5,
        max_value=5.0,
        value=4.0,
        step=0.5,
        help="0.5 = Ruim, 5.0 = Excelente",
        key=f"rating_slider_{movie_id}",
    )

    st.info(f"Nota selecionada: {rating:.1f}/5")

    col1, col2 = st.columns(2)
    if col1.button("Salvar avaliação", type="primary", use_container_width=True):
        backend_url = os.getenv("BACKEND_URL")

        if not backend_url:
            # tenta ler de secrets **apenas se existir**
            try:
                backend_url = st.secrets["backend"]["url"]
            except Exception:
                backend_url = None

        if not backend_url:
            st.error("BACKEND_URL não configurado (.env ou .streamlit/secrets.toml).")
            st.stop()
        payload = {
            "user_id": int(user_id),
            "movie_id": int(movie_id),
            "rating": float(rating),
        }

        try:
            with st.spinner("Salvando..."):
                resp = requests.post(
                    f"{backend_url}/api/ratings",
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()
            st.success("Avaliação salva com sucesso!")
            st.cache_data.clear()
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Falha ao salvar avaliação: {e}")

    if col2.button("Cancelar", use_container_width=True):
        st.rerun()


# -------- modal de detalhes --------
@st.dialog("Detalhes do filme")
def show_movie_details(movie_id: int, fallback_row: pd.Series | None = None):
    """
    Exibe detalhes do filme.

    Tenta buscar no backend em GET /api/{movie_id}.
    Se falhar, usa as informações do DataFrame (fallback_row).
    """
    backend_url = os.getenv("BACKEND_URL")
    if not backend_url:
        try:
            backend_url = st.secrets["backend"]["url"]
        except Exception:
            backend_url = None

    movie_data = None

    if backend_url:
        try:
            backend_url = backend_url.rstrip("/")
            resp = requests.get(f"{backend_url}/api/{movie_id}", timeout=30)
            resp.raise_for_status()
            movie_data = resp.json()
        except requests.RequestException as e:
            st.warning(f"Não foi possível carregar detalhes do backend: {e}")

    # fallback: usa dados do DF
    if movie_data is None and fallback_row is not None:
        movie_data = fallback_row.to_dict()

    if movie_data is None:
        st.error("Detalhes não disponíveis para este filme.")
        return

    title = movie_data.get("title") or (fallback_row.get("title") if fallback_row is not None else "—")
    genres = movie_data.get("genres") or (fallback_row.get("genres") if fallback_row is not None else None)
    overview = movie_data.get("overview") or "Sinopse não disponível."
    release_date = movie_data.get("release_date") or (
        fallback_row.get("release_date") if fallback_row is not None else None
    )
    popularity = movie_data.get("popularity") or (
        fallback_row.get("popularity") if fallback_row is not None else None
    )
    companies = movie_data.get("companies_text") or movie_data.get("production_companies")

    poster_path = movie_data.get("poster_path") or (
        fallback_row.get("poster_path") if fallback_row is not None else None
    )
    poster_url = _build_poster_url(poster_path)

    st.markdown(f"### {title}")
    st.markdown("---")

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(poster_url, width="stretch")
    with col_info:
        if genres:
            st.caption(f"🎭 {genres}")
        if release_date:
            year = str(release_date)[:4]
            st.caption(f"📅 {year}")
        if companies:
            st.caption(f"🏢 {companies}")
        if popularity:
            st.caption(f"🔥 popularity: {popularity:.2f}")

    st.markdown("#### Sinopse")
    st.write(overview)


def render_movie_card(row: pd.Series) -> None:
    """Card de filme no catálogo."""
    with st.container():
        poster_url = _build_poster_url(row.get("poster_path"))
        st.image(poster_url, width="stretch")

        title = row.get("title", "Sem título")
        st.markdown(f"**{title}**")

        if pd.notna(row.get("genres")):
            st.caption(f"🎭 {row['genres']}")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            rel = row.get("release_date")
            if pd.notna(rel):
                year = str(rel)[:4]
                st.caption(f"📅 {year}")
        with col2:
            mid = row.get("id")
            if pd.notna(mid):
                st.caption(f"ID: {int(mid)}")
        with col3:
            key_suffix = row.get("id", row.name)
            b1, b2 = st.columns(2)
            with b1:
                if st.button(
                    "⭐ Avaliar",
                    key=f"rate_{key_suffix}",
                    type="secondary",
                    use_container_width=True,
                ):
                    show_rating_dialog(int(row["id"]), title)
            with b2:
                if st.button(
                    "Detalhes",
                    key=f"details_{key_suffix}",
                    type="secondary",
                    use_container_width=True,
                ):
                    show_movie_details(int(row["id"]), row)


def render(movies_df: DataFrame) -> None:
    """Página de catálogo de filmes."""
    st.subheader("🎬 Catálogo de Filmes")

    search: str = st.text_input(
        "🔍 Buscar filme ou gênero:",
        placeholder="Digite o nome do filme ou gênero...",
    )

    df: DataFrame = movies_df.copy()

    if search:
        text = search.strip().lower()
        mask = (
            df["title"].astype(str).str.contains(text, case=False, na=False)
            | df["genres"].astype(str).str.contains(text, case=False, na=False)
        )
        df = df[mask]

        if df.empty:
            st.warning("Nenhum filme encontrado.")
            return

    df = (
        df.sort_values(["title", "poster_path"], ascending=[True, False])
        .drop_duplicates(subset=["title"])
    )
    page: DataFrame = paginate(df, page_size=9, key="catalog_page")

    cols = st.columns(3)
    for idx, (_, row) in enumerate(page.iterrows()):
        with cols[idx % 3]:
            render_movie_card(row)

    st.divider()
