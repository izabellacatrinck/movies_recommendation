from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from pandas import DataFrame


def paginate(df: DataFrame, page_size: int, key: str) -> DataFrame:
    total: int = len(df)
    if total == 0:
        return df

    if key not in st.session_state:
        st.session_state[key] = 1

    page = st.session_state[key]
    n_pages: int = max(1, (total + page_size - 1) // page_size)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Anterior", disabled=(page <= 1), key=f"{key}_prev"):
            st.session_state[key] = page - 1
            st.rerun()

    with col2:
        st.markdown(f"<div style='text-align: center'>Página {page}/{n_pages}</div>", unsafe_allow_html=True)

    with col3:
        if st.button("Próxima →", disabled=(page >= n_pages), key=f"{key}_next"):
            st.session_state[key] = page + 1
            st.rerun()
    st.caption(f"Total: {total} itens")

    # Retorna slice da página atual
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]