from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from pandas import DataFrame

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"


def _require(p: Path):
    if not p.exists():
        st.error(f"Arquivo ausente: {p}")
        st.stop()


@st.cache_data
def load_data() -> tuple[DataFrame, DataFrame]:
    ratings_path: Path = DATA_DIR / "ratings_final_df.csv"
    movies_path: Path = DATA_DIR / "movies_final_df.csv"

    _require(ratings_path)
    _require(movies_path)

    ratings_df: DataFrame = pd.read_csv(ratings_path)
    movies_df: DataFrame = pd.read_csv(movies_path)

    return ratings_df, movies_df
