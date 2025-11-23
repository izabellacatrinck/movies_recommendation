from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq
from data import load_data
from dotenv import load_dotenv

load_dotenv()


def _get_groq_api_key() -> str | None:
    """Tenta pegar a GROQ_API_KEY do .env ou do st.secrets."""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key

    try:
        api_key = st.secrets["groq"]["api_key"]
    except Exception:
        api_key = None

    return api_key


def search_movies_in_csv(query: str, max_results: int = 30) -> str:
    """
    Busca filmes no CSV de catálogo (movies_df) usando várias colunas
    e retorna um JSON (lista de objetos) com, no máximo, `max_results` itens.
    """
    _, df_movies = load_data()

    # colunas que vamos usar para match de texto
    search_cols: list[str] = [
        "title",
        "original_title",
        "genres",
        "overview",
        "production_companies",
    ]

    # garante que as colunas existem e são string
    for col in search_cols:
        if col not in df_movies.columns:
            df_movies[col] = ""
        df_movies[col] = df_movies[col].astype(str)

    mask = np.logical_or.reduce(
        [df_movies[col].str.contains(query, case=False, na=False) for col in search_cols]
    )

    results_df = df_movies[mask]

    if results_df.empty:
    # fallback: pega os mais populares do catálogo inteiro
        if "popularity" in df_movies.columns:
            results_df = df_movies.sort_values("popularity", ascending=False).head(max_results)
        else:
            results_df = df_movies.head(max_results)

    # seleciona algumas colunas úteis pra mandar pro modelo
    cols_out = [
        c
        for c in [
            "id",
            "title",
            "original_title",
            "genres",
            "overview",
            "vote_average",
            "popularity",
        ]
        if c in results_df.columns
    ]
    results_df = results_df[cols_out].head(max_results)

    return results_df.to_json(orient="records", force_ascii=False)


def get_user_rated_movies_json(user_id: str | int, max_results: int = 30) -> str:
    """
    Retorna, em JSON, os filmes avaliados pelo usuário (juntando ratings + movies).
    """
    ratings_df, movies_df = load_data()

    hist = ratings_df[ratings_df["userId"].astype(str) == str(user_id)].copy()
    if hist.empty:
        return "[]"

    hist = hist.merge(
        movies_df,
        left_on="movieId",
        right_on="id",
        how="left",
    )

    # remove linhas claramente inválidas
    hist = hist[
        hist["movieId"].notna()
        & hist["rating"].notna()
    ]
    hist["movieId"] = pd.to_numeric(hist["movieId"], errors="coerce")
    hist["rating"] = pd.to_numeric(hist["rating"], errors="coerce")
    hist = hist[(hist["movieId"] > 0) & (hist["rating"] > 0)]

    if hist.empty:
        return "[]"

    cols_out = [
        c
        for c in [
            "id",
            "title",
            "genres",
            "vote_average",
            "rating",
        ]
        if c in hist.columns
    ]
    hist = hist[cols_out].head(max_results)

    return hist.to_json(orient="records", force_ascii=False)


@st.cache_resource
def setup_agent() -> Agent | None:
    """
    Cria e guarda em cache o Agent da Agno usando Groq.
    (Sem ferramentas declaradas — nós mesmos passamos o contexto no prompt.)
    """
    groq_api_key = _get_groq_api_key()
    if not groq_api_key:
        return None

    agent = Agent(
        name="movie_agent",
        role=(
            "Você é um cinéfilo especialista em filmes. "
            "Use apenas as informações de filmes fornecidas no contexto do prompt "
            "para responder. Responda sempre em português. "
            "Se o usuário perguntar sobre outro assunto que não seja filmes, diga "
            "que não tem conhecimento no assunto."
        ),
        model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
        markdown=True,
    )
    return agent


def _build_augmented_prompt(user_prompt: str) -> str:
    """
    Monta um prompt enriquecido com:
    - filmes do catálogo relacionados ao texto da pergunta;
    - filmes avaliados pelo usuário atual (se houver).
    """
    current_user = st.session_state.get("current_user")

    # busca geral no catálogo
    catalog_json = search_movies_in_csv(user_prompt, max_results=40)

    # filmes avaliados pelo usuário
    if current_user:
        user_json = get_user_rated_movies_json(current_user, max_results=40)
    else:
        user_json = "[]"

    prompt = f"""
Você é um especialista em filmes.

Abaixo você tem dados em JSON sobre filmes do catálogo
e também, se disponível, os filmes avaliados pelo usuário atual.

Use APENAS essas informações para responder em português à pergunta do usuário.
Se algo não estiver nos dados, admita que não sabe.

=== CATÁLOGO DE FILMES RELACIONADOS À BUSCA ===
{catalog_json}

=== FILMES AVALIADOS PELO USUÁRIO ATUAL (id={current_user or "desconhecido"}) ===
{user_json}

Agora responda à pergunta do usuário de forma clara, em português,
podendo citar títulos, gêneros, popularidade ou nota média
quando fizer sentido.

Pergunta do usuário:
\"\"\"{user_prompt}\"\"\""""

    return prompt


def render():
    st.subheader("🤖 Converse com o Agente Cinéfilo")

    movie_agent = setup_agent()
    if not movie_agent:
        st.error(
            "Chave da API do Groq não configurada. "
            "Verifique seu arquivo .env (GROQ_API_KEY) ou os segredos do Streamlit."
        )
        return

    # inicializa memória do chat se ainda não existir
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []

    with st.form(key="chat_form", clear_on_submit=True):
        prompt = st.text_input(
            "Peça uma recomendação ou informação...",
            placeholder="Ex: Quais os filmes mais populares do gênero romance?",
            label_visibility="collapsed",
        )
        submit_button = st.form_submit_button("Enviar Pergunta")

    if submit_button and prompt:
        # registra pergunta do usuário
        st.session_state.agent_messages.append({"role": "user", "content": prompt})

        # monta prompt enriquecido
        augmented_prompt = _build_augmented_prompt(prompt)

        # gera resposta
        with st.spinner("O agente está pensando..."):
            run_output = movie_agent.run(augmented_prompt)
            response_text = run_output.content

        # registra resposta
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": response_text}
        )

    st.divider()

    # exibe da mais nova para a mais antiga
    for message in reversed(st.session_state.agent_messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
