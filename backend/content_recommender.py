from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ContentKNNConfig:
    movies_csv_path: str = "data/movies_final_df.csv"
    ratings_csv_path: str = "data/ratings_final_df.csv"
    content_column: str = "content"
    movie_id_column: str = "id"         # coluna de ID em movies
    ratings_user_col: str = "userId"    # coluna de usuário em ratings
    ratings_movie_col: str = "movieId"  # coluna de filme em ratings
    ratings_score_col: str = "rating"
    max_features: Optional[int] = 10000
    stop_words: Optional[str] = "english"
    knn_neighbors: int = 50


class ContentKNNRecommender:
    """
    Recomendador híbrido baseado em conteúdo + item–item colaborativo.

    - TF-IDF em cima da coluna `content` do movies_final_df (já pré-montada).
    - kNN item–item em espaço de conteúdo (similar_movies / fallback).
    - Similaridade item–item colaborativa em cima da matriz usuário x filme.
    - Recomendações para usuário: combinação de
        - perfil de conteúdo do usuário (vetor médio ponderado centrado)
        - filtro colaborativo item–item (predição baseada em ratings do usuário)
    """

    def __init__(self, config: ContentKNNConfig | None = None):
        self.config = config or ContentKNNConfig()

        # Data
        self.df_movies: Optional[pd.DataFrame] = None
        self.df_ratings: Optional[pd.DataFrame] = None

        # Vetorização / modelos de conteúdo
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X_movies = None  # matriz TF-IDF (num_filmes x num_features)
        self.knn_model: Optional[NearestNeighbors] = None

        # Mapas de índice <-> movie_id
        self.idx_to_movie_id: List[int] = []
        self.movie_id_to_idx: Dict[int, int] = {}

        # Estruturas colaborativas (item–item por ratings)
        self.user_ids: List[int] = []
        self.user_to_idx: Dict[int, int] = {}
        self.R: Optional[np.ndarray] = None          # matriz (num_filmes x num_users)
        self.item_sim_cf: Optional[np.ndarray] = None  # matriz de similaridade item–item

    # ------------- CARREGAMENTO E TREINAMENTO -------------

    def load_data(self) -> None:
        """Carrega CSVs de filmes e ratings e faz alinhamento básico."""
        cfg = self.config

        if not os.path.exists(cfg.movies_csv_path):
            raise FileNotFoundError(f"Movies CSV não encontrado: {cfg.movies_csv_path}")
        if not os.path.exists(cfg.ratings_csv_path):
            raise FileNotFoundError(f"Ratings CSV não encontrado: {cfg.ratings_csv_path}")

        self.df_movies = pd.read_csv(cfg.movies_csv_path)
        self.df_ratings = pd.read_csv(cfg.ratings_csv_path)

        # normalizar nomes das colunas de ratings para interno
        self.df_ratings = self.df_ratings.rename(
            columns={
                cfg.ratings_user_col: "user_id",
                cfg.ratings_movie_col: "movie_id",
                cfg.ratings_score_col: "rating",
            }
        )

        # garantir tipos
        self.df_movies[cfg.movie_id_column] = self.df_movies[cfg.movie_id_column].astype(int)
        self.df_ratings["movie_id"] = self.df_ratings["movie_id"].astype(int)
        self.df_ratings["user_id"] = self.df_ratings["user_id"].astype(int)

        # manter apenas filmes em comum entre movies e ratings
        movies_in_movies = set(self.df_movies[cfg.movie_id_column].unique())
        movies_in_ratings = set(self.df_ratings["movie_id"].unique())
        common = movies_in_movies & movies_in_ratings

        self.df_movies = (
            self.df_movies[self.df_movies[cfg.movie_id_column].isin(common)]
            .reset_index(drop=True)
        )
        self.df_ratings = (
            self.df_ratings[self.df_ratings["movie_id"].isin(common)]
            .reset_index(drop=True)
        )

    def _build_mappings(self) -> None:
        """Cria mapas índice <-> movie_id alinhados com X_movies."""
        cfg = self.config
        movie_ids = self.df_movies[cfg.movie_id_column].astype(int).tolist()
        self.idx_to_movie_id = movie_ids
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}

    @staticmethod
    def _split_genres(genres_val) -> List[str]:
        """Normaliza string de gêneros em lista de tokens."""
        if pd.isna(genres_val):
            return []
        return str(genres_val).lower().replace("|", " ").split()

    def _get_user_top_genres(
        self,
        user_id: int,
        min_rating_for_genres: float = 4.0,
        top_n: int = 5,
        min_occurrences: int = 3,
    ) -> List[str]:
        """
        Retorna os top N gêneros que o usuário mais consome,
        ponderados pela nota, considerando apenas gêneros que aparecem
        em pelo menos `min_occurrences` likes do usuário.
        Usado mais para debug/inspeção do que para ranking direto.
        """
        user_ratings = self.df_ratings[self.df_ratings["user_id"] == user_id]

        liked = user_ratings[user_ratings["rating"] >= min_rating_for_genres].copy()
        if liked.empty:
            return []

        df = liked.merge(
            self.df_movies[[self.config.movie_id_column, "genres"]],
            left_on="movie_id",
            right_on=self.config.movie_id_column,
            how="left",
        )

        genre_scores: Dict[str, float] = {}
        genre_counts: Dict[str, int] = {}

        for _, row in df.iterrows():
            rating = float(row["rating"])
            genres_list = self._split_genres(row["genres"])
            for g in genres_list:
                genre_counts[g] = genre_counts.get(g, 0) + 1
                genre_scores[g] = genre_scores.get(g, 0.0) + rating

        if not genre_scores:
            return []

        filtered_scores = {
            g: score
            for g, score in genre_scores.items()
            if genre_counts.get(g, 0) >= min_occurrences
        }

        if not filtered_scores:
            filtered_scores = genre_scores

        series_scores = pd.Series(filtered_scores).sort_values(ascending=False)
        return series_scores.head(top_n).index.tolist()

    def _build_cf_matrices(self) -> None:
        """
        Constrói:
        - matriz R (num_filmes x num_users) com ratings
        - matriz de similaridade item–item colaborativa (cosine em R centrado por usuário)
        """
        df = self.df_ratings
        movie_ids = self.idx_to_movie_id
        user_ids = sorted(df["user_id"].unique().tolist())

        self.user_ids = user_ids
        self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

        n_movies = len(movie_ids)
        n_users = len(user_ids)

        if n_movies == 0 or n_users == 0:
            self.R = None
            self.item_sim_cf = None
            return

        R = np.zeros((n_movies, n_users), dtype=np.float32)

        # preencher ratings
        for row in df.itertuples(index=False):
            mid = int(row.movie_id)
            uid = int(row.user_id)
            rating = float(row.rating)

            if mid not in self.movie_id_to_idx:
                continue
            if uid not in self.user_to_idx:
                continue

            i = self.movie_id_to_idx[mid]
            j = self.user_to_idx[uid]
            R[i, j] = rating

        # centralizar por usuário (remove viés de "usuário que dá nota alta/baixa")
        R_centered = R.copy()
        for j in range(n_users):
            col = R[:, j]
            mask = col > 0
            if mask.any():
                mean = col[mask].mean()
                R_centered[mask, j] = col[mask] - mean

        # similaridade item–item colaborativa
        self.item_sim_cf = cosine_similarity(R_centered)
        self.R = R

    def fit(self) -> None:
        """Treina o TF-IDF + kNN de conteúdo + estruturas colaborativas."""
        if self.df_movies is None or self.df_ratings is None:
            self.load_data()

        cfg = self.config
        content_col = cfg.content_column

        if content_col not in self.df_movies.columns:
            raise ValueError(f"Coluna '{content_col}' não encontrada em df_movies.")

        # Usar o content já pré-processado no CSV final
        text = (
            self.df_movies[content_col]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        self.vectorizer = TfidfVectorizer(
            stop_words=cfg.stop_words,
            max_features=cfg.max_features or 20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.6,      # ignora termos muito frequentes (quase stopwords de domínio)
            sublinear_tf=True,
        )
        self.X_movies = self.vectorizer.fit_transform(text)

        # mapas índice <-> movie_id
        self._build_mappings()

        # Modelo kNN item–item (conteúdo)
        self.knn_model = NearestNeighbors(
            n_neighbors=cfg.knn_neighbors,
            metric="cosine",
            algorithm="brute",
        )
        self.knn_model.fit(self.X_movies)

        # Estruturas colaborativas
        self._build_cf_matrices()

    # ------------- ITEM-ITEM: FILMES PARECIDOS (CONTEÚDO) -------------

    def similar_movies(self, movie_id: int, top_n: int = 10) -> pd.DataFrame:
        """
        Retorna filmes mais parecidos com um filme específico, usando kNN item–item
        em espaço de conteúdo.
        """
        if self.knn_model is None or self.X_movies is None:
            self.fit()

        if movie_id not in self.movie_id_to_idx:
            raise ValueError(f"movie_id {movie_id} não encontrado em df_movies.")

        idx = self.movie_id_to_idx[movie_id]
        movie_vec = self.X_movies[idx]

        distances, indices = self.knn_model.kneighbors(movie_vec, n_neighbors=top_n + 1)
        distances = distances.flatten()
        indices = indices.flatten()

        # primeiro vizinho é o próprio filme (dist=0) -> descarta
        indices = indices[1 : top_n + 1]
        distances = distances[1 : top_n + 1]

        scores = 1.0 - distances  # cosine distance -> similarity

        sim_ids = [self.idx_to_movie_id[i] for i in indices]

        df_sim = pd.DataFrame({
            "movie_id": sim_ids,
            "similarity": scores,
        })
        df_sim = df_sim.merge(
            self.df_movies,
            left_on="movie_id",
            right_on=self.config.movie_id_column,
            how="left",
        )

        return df_sim[["movie_id", "title", "similarity", "genres"]].reset_index(drop=True)

    # ------------- USER-BASED: PERFIL DE USUÁRIO (CONTEÚDO) -------------

    def _build_user_profile(self, user_id: int):
        """
        Perfil de usuário em conteúdo.
        - Usa TODAS as notas do usuário para os filmes presentes em df_movies.
        - Peso de cada filme = (rating - média_do_usuário).
          > filmes acima da média puxam o vetor na direção deles;
          > filmes abaixo da média empurram para longe (peso negativo).
        """
        user_ratings = self.df_ratings[self.df_ratings["user_id"] == user_id]
        if user_ratings.empty:
            return None

        rows = []
        for _, row in user_ratings.iterrows():
            mid = int(row["movie_id"])
            rating = float(row["rating"])
            if mid not in self.movie_id_to_idx:
                continue
            idx = self.movie_id_to_idx[mid]
            rows.append((idx, rating))

        if not rows:
            return None

        indices = np.array([r[0] for r in rows], dtype=int)
        ratings = np.array([r[1] for r in rows], dtype=float)

        mean_rating = ratings.mean()
        weights = ratings - mean_rating  # pode ser negativo

        # tira pesos muito pequenos (ruído)
        MIN_ABS_WEIGHT = 0.2
        mask = np.abs(weights) > MIN_ABS_WEIGHT
        if not mask.any():
            return None

        indices = indices[mask]
        weights = weights[mask].reshape(-1, 1)

        X_liked = self.X_movies[indices]
        profile_mat = (X_liked.multiply(weights)).sum(axis=0)

        profile_vec = np.asarray(profile_mat).reshape(1, -1)
        norm = np.linalg.norm(profile_vec)
        if norm > 0:
            profile_vec = profile_vec / norm

        return profile_vec

    # ------------- USER-BASED: SCORES COLABORATIVOS -------------

    def _cf_scores_for_user(self, user_id: int) -> Optional[np.ndarray]:
        """
        Retorna vetor de scores colaborativos item–item para todos os filmes
        (mesma ordem de idx_to_movie_id). Se não for possível, retorna None.
        """
        if self.item_sim_cf is None or self.R is None or not self.user_to_idx:
            return None
        if user_id not in self.user_to_idx:
            return None

        j = self.user_to_idx[user_id]
        user_ratings = self.R[:, j].astype(np.float32)
        mask = user_ratings > 0
        if not mask.any():
            return None

        mean_u = user_ratings[mask].mean()
        r_centered = np.zeros_like(user_ratings)
        r_centered[mask] = user_ratings[mask] - mean_u

        # predição colaborativa: S * r_centered / sum(|S| * 1_mask)
        numer = self.item_sim_cf.dot(r_centered)
        denom = np.abs(self.item_sim_cf).dot((r_centered != 0).astype(np.float32))
        denom[denom == 0] = 1e-6

        scores = numer / denom
        return scores

    # ------------- USER COLD-START: ITEM-ITEM CONTEÚDO -------------

    def _recommend_from_neighbors(
        self,
        user_likes: pd.DataFrame,
        watched_ids: set[int],
        top_n: int = 10,
        per_movie_neighbors: int = 20,
    ) -> pd.DataFrame:
        """
        Fallback bem simples:
        - Para usuários com pouquíssimo histórico, pega vizinhos de cada filme curtido
          no espaço de conteúdo e soma similaridades.
        """
        if self.knn_model is None or self.X_movies is None:
            self.fit()

        candidate_scores: Dict[int, float] = {}

        for _, row in user_likes.iterrows():
            mid = int(row["movie_id"])
            rating = float(row["rating"])

            if mid not in self.movie_id_to_idx:
                continue

            idx = self.movie_id_to_idx[mid]
            movie_vec = self.X_movies[idx]

            distances, indices = self.knn_model.kneighbors(
                movie_vec, n_neighbors=per_movie_neighbors + 1
            )
            distances = distances.flatten()
            indices = indices.flatten()

            # descarta o próprio filme
            indices = indices[1:]
            distances = distances[1:]

            sims = 1.0 - distances

            # peso simples pelo rating (não centrado aqui, é só fallback)
            weight = max(rating - 3.0, 0.5)

            for i, sim in zip(indices, sims):
                cand_mid = self.idx_to_movie_id[i]
                if cand_mid in watched_ids:
                    continue
                candidate_scores[cand_mid] = candidate_scores.get(cand_mid, 0.0) + float(sim) * weight

        if not candidate_scores:
            return pd.DataFrame()

        df_cand = (
            pd.DataFrame(
                [{"movie_id": m, "score": s} for m, s in candidate_scores.items()]
            )
            .sort_values("score", ascending=False)
            .head(top_n)
        )

        df_cand = df_cand.merge(
            self.df_movies[[self.config.movie_id_column, "title", "genres"]],
            left_on="movie_id",
            right_on=self.config.movie_id_column,
            how="left",
        )

        return df_cand[["movie_id", "title", "score", "genres"]].reset_index(drop=True)

    # ------------- RECOMENDAÇÃO PARA USUÁRIO (HÍBRIDO) -------------

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        min_rating: float = 4.0,  # mantido para compatibilidade com test_recommender (não é mais crucial)
    ) -> pd.DataFrame:
        """
        Recomenda filmes para um usuário.

        Estratégia:
        - Se o usuário tiver pouquíssimos ratings (ex: < 5), usa apenas fallback
          item–item em conteúdo.
        - Caso contrário:
            * monta perfil de conteúdo do usuário (todas as notas, centradas)
            * pega scores colaborativos item–item
            * combina score de conteúdo + score colaborativo
        """
        if self.knn_model is None or self.X_movies is None:
            self.fit()

        user_ratings = self.df_ratings[self.df_ratings["user_id"] == user_id]
        if user_ratings.empty:
            return pd.DataFrame()

        watched_ids = set(user_ratings["movie_id"].astype(int).tolist())

        # COLD-START bem agressivo: pouquíssimos ratings -> só neighbors de conteúdo
        if len(user_ratings) < 5:
            user_likes_loose = user_ratings[user_ratings["rating"] >= 3.0].copy()
            if user_likes_loose.empty:
                return pd.DataFrame()
            return self._recommend_from_neighbors(
                user_likes=user_likes_loose,
                watched_ids=watched_ids,
                top_n=top_n,
                per_movie_neighbors=20,
            )

        # Perfil de conteúdo do usuário
        user_profile = self._build_user_profile(user_id)
        if user_profile is not None:
            sim_scores = cosine_similarity(user_profile, self.X_movies).flatten()
        else:
            sim_scores = np.zeros(self.X_movies.shape[0], dtype=np.float32)

        # Scores colaborativos para o usuário (pode ser None)
        cf_scores = self._cf_scores_for_user(user_id)
        if cf_scores is None:
            cf_scores = np.zeros(len(self.idx_to_movie_id), dtype=np.float32)
        else:
            cf_scores = np.asarray(cf_scores, dtype=np.float32)

        # Monta dataframe de scores para todos os filmes
        df_scores = pd.DataFrame({
            "movie_id": self.df_movies[self.config.movie_id_column].astype(int),
            "score_content": sim_scores,
            "score_cf": cf_scores,
        })

        # remove filmes já vistos
        df_scores = df_scores[~df_scores["movie_id"].isin(watched_ids)]

        # junta metadados de filme
        df_scores = df_scores.merge(
            self.df_movies[[self.config.movie_id_column, "title", "genres"]],
            left_on="movie_id",
            right_on=self.config.movie_id_column,
            how="left",
        )

        # combinação simples de conteúdo + CF
        ALPHA = 0.6  # peso do conteúdo
        BETA = 0.4   # peso do colaborativo
        df_scores["score"] = ALPHA * df_scores["score_content"] + BETA * df_scores["score_cf"]

        df_top = df_scores.sort_values("score", ascending=False).head(top_n)

        return df_top[["movie_id", "title", "score", "genres"]].reset_index(drop=True)


if __name__ == "__main__":
    cfg = ContentKNNConfig()
    recommender = ContentKNNRecommender(cfg)
    recommender.fit()

    first_movie_id = int(recommender.df_movies.iloc[0][recommender.config.movie_id_column])
    print("Filme base:", first_movie_id, "-", recommender.df_movies.iloc[0]["title"])
    print("\nFilmes similares (conteúdo):")
    print(recommender.similar_movies(first_movie_id, top_n=5))

    some_user = int(recommender.df_ratings["user_id"].unique()[0])
    print(f"\nRecomendações híbridas para usuário {some_user}:")
    print(recommender.recommend_for_user(some_user, top_n=5))
