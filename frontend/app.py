from __future__ import annotations

import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent 
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import requests
from dotenv import load_dotenv

from data import load_data
from app_pages.catalog import render as render_catalog 
from app_pages.ratings import render as render_ratings
from app_pages.recommender import render as render_recommendations
from app_pages.agent import render as render_agent
import state

st.set_page_config(page_title="Movie Recommender", layout="wide")

load_dotenv()


st.markdown(
    """
    <style>
      img[data-testid="stImage"] {
        object-fit: contain;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def sidebar(user_ids) -> None:
    with st.sidebar:
        st.header("Conta")

        flash = st.session_state.pop("flash", None)
        if flash:
            kind, msg = flash.get("type"), flash.get("msg", "")
            if kind == "success":
                st.success(msg)
            elif kind == "warning":
                st.warning(msg)
            elif kind == "error":
                st.error(msg)
            else:
                st.info(msg)

        current = st.session_state.get("current_user")
        st.caption(f"Logado como: **{current or '—'}**")

        st.divider()

        options: list[str] = [str(x) for x in user_ids]
        default_idx = 0
        if current and str(current) in options:
            default_idx = options.index(str(current))

        uid: str = st.selectbox(
            "Selecionar usuário existente",
            options=options or ["—"],
            index=default_idx if options else 0,
        )
        if st.button("Entrar", use_container_width=True, disabled=not options):
            st.session_state.current_user = str(uid)
            st.toast(f"Logado como {uid}")

        st.divider()

        new_id = st.text_input("Cadastrar novo ID", value="", placeholder="ex.: 9999")
        if st.button("Cadastrar", use_container_width=True):
            new_uid = new_id.strip()
            if not new_uid:
                st.warning("Informe um ID.")
            else:
                try:

                    backend_url = os.getenv("BACKEND_URL")
                    if not backend_url:
                        try:
                            backend_url = st.secrets["backend"]["url"]
                        except Exception:
                            backend_url = None

                    if not backend_url:
                        st.error("BACKEND_URL não configurado (.env ou .streamlit/secrets.toml).")
                        st.stop()

                    payload = {"id": int(new_uid)}

                    r = requests.post(
                        f"{backend_url}/api/users",
                        json=payload,
                        timeout=60,
                    )
                    r.raise_for_status()

                    try:
                        from data import load_data  # se já não estiver no topo
                        load_data.clear()
                    except Exception:
                        st.cache_data.clear()

                    st.session_state["flash"] = {
                        "type": "success",
                        "msg": f"Usuário **{new_uid}** cadastrado com sucesso.",
                    }
                    st.session_state["current_user"] = new_uid
                    st.rerun()

                except ValueError:
                    st.warning("O ID deve ser numérico (ex.: 9999).")
                except requests.RequestException as e:
                    st.error(f"Falha ao cadastrar: {e}")



def main():
    state.init()

    ratings_df, movies_df = load_data()

    user_ids = (
        ratings_df["userId"]
        .dropna()
        .astype(int)
        .sort_values()
        .astype(str)
        .unique()
    )

    sidebar(user_ids)

    st.markdown("# 🎞️ Sistema de Recomendação de Filmes")
    st.caption(
        "Frontend em Streamlit consumindo o backend em FastAPI "
        "com recomendações baseadas em conteúdo e filtragem colaborativa."
    )

    tab_cat, tab_my, tab_rec, tab_agent = st.tabs(
        ["📚 Catálogo", "⭐ Minhas avaliações", "✨ Recomendações", "🤖 Agente"]
    )

    with tab_cat:
        render_catalog(movies_df)

    with tab_my:
        render_ratings(ratings_df, movies_df)

    with tab_rec:
        render_recommendations(ratings_df, movies_df)

    with tab_agent:
        render_agent() 


if __name__ == "__main__":
    main()
