# STRUTTURA: CORPUS -> CANDIDATE GENERATION -> SCORING -> RE-RANKING -> FINAL RANKING
#
# Gestione cold-start
# -------------------
# UTENTI NUOVI (non in U):
#   Fold-in: si calcola x_u al volo dalla history.
#
# ARTICOLI NUOVI (non in V al training):
#   Joint index per news_id nel training.
#
# --- Due modalità di evaluation -------------------------------------------
#
# RANKING GLOBALE (per la demo, top_k su tutte le news):
#   recommend_for_user(..., impression_pool=None)
#   Usato per mostrare raccomandazioni all'utente finale.
#   Esclude gli item già visti nel training.
#
# RANKING SUL POOL DI IMPRESSION (per le metriche):
#   recommend_for_user(..., impression_pool=[...])
#   la valutazione avviene sul sottoinsieme di news mostrate in quella
#   sessione, non su tutte le 65k news del corpus.

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp

from typing import List, Dict, Tuple, Optional


class NewsRecommenderSystem:

    def __init__(
        self,
        artifacts_dir: str,
        popularity_csv_path: str = "data/train/popularity_data.csv",
        popularity_weight: float = 0.05, 
        alpha: float = 40.0,
        reg: float = 0.01,
    ):
        self.artifacts_dir = artifacts_dir
        self.popularity_csv_path = popularity_csv_path
        self.popularity_weight = popularity_weight
        self.alpha = alpha
        self.alpha_neg = 20.0   # confidenza per impression non cliccate 
        self.reg = reg

        self.U: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.news_df: Optional[pd.DataFrame] = None
        self.C_pos: Optional[sp.csr_matrix] = None
        self.popularity_dict: Dict[str, float] = {}

        self._initialize_system()

    # ------------------------------------------------------------------
    # Inizializzazione
    # ------------------------------------------------------------------

    def _initialize_system(self):
        print("[Recommender] Inizializzazione...")

        # 1. mappings
        index_path = os.path.join(self.artifacts_dir, "index.npz")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"index.npz non trovato in {self.artifacts_dir}")
        data = np.load(index_path, allow_pickle=True)
        self.idx2user = data["idx2user"]
        self.idx2news = data["idx2news"]
        self.user2idx = {u: i for i, u in enumerate(self.idx2user)}
        self.news2idx = {n: i for i, n in enumerate(self.idx2news)}

        # 2. fattori latenti U, V
        u_path = os.path.join(self.artifacts_dir, "U.npy")
        v_path = os.path.join(self.artifacts_dir, "V.npy")
        if not (os.path.exists(u_path) and os.path.exists(v_path)):
            raise FileNotFoundError("U.npy o V.npy non trovati. Esegui prima train.py.")
        self.U = np.load(u_path)
        self.V = np.load(v_path)
        self.n_factors = self.V.shape[1]

        # 3. corpus: parquet o tsv
        parquet_path = os.path.join(self.artifacts_dir, "news.parquet")
        if os.path.exists(parquet_path):
            self.news_df = pd.read_parquet(parquet_path)
            print(f"[Corpus] {len(self.news_df)} notizie da news.parquet.")
        else:
            for tsv_candidate in [
                os.path.join(os.path.dirname(self.artifacts_dir), "data", "train", "news.tsv"),
                "data/train/news.tsv",
            ]:
                if os.path.exists(tsv_candidate):
                    NEWS_COLS = [
                        "news_id", "category", "subcategory", "title",
                        "abstract", "url", "title_entities", "abstract_entities",
                    ]
                    self.news_df = pd.read_table(
                        tsv_candidate, header=None, names=NEWS_COLS, na_filter=False
                    )
                    print(f"[Corpus] {len(self.news_df)} notizie da news.tsv.")
                    break
            else:
                print("[Warning] news.parquet / news.tsv non trovati.")

        # 4. popularity
        self._load_popularity_data()

        # 5. C_pos (escludendo item di training già visti)
        c_pos_path = os.path.join(self.artifacts_dir, "C_pos.npz")
        if os.path.exists(c_pos_path):
            self.C_pos = sp.load_npz(c_pos_path)
        else:
            print("[Warning] C_pos.npz non trovato: filtraggio item-visti disabilitato.")

        print(
            f"[Recommender] Pronto: {len(self.idx2user)} utenti, "
            f"{len(self.idx2news)} news, {self.n_factors} fattori."
        )

    def _load_popularity_data(self):
        if not os.path.exists(self.popularity_csv_path):
            print(f"[Warning] Popolarità non trovata: {self.popularity_csv_path}")
            return
        pop_df = pd.read_csv(self.popularity_csv_path)
        if "article_id" in pop_df.columns and "clicks" in pop_df.columns:
            lo, hi = pop_df["clicks"].min(), pop_df["clicks"].max()
            pop_df["norm_score"] = (
                (pop_df["clicks"] - lo) / (hi - lo) if hi > lo else 1.0
            )
            self.popularity_dict = dict(zip(pop_df["article_id"], pop_df["norm_score"]))
            print(f"[Popularity] {len(self.popularity_dict)} articoli caricati.")

    # ------------------------------------------------------------------
    # vettore utente
    # ------------------------------------------------------------------

    def _get_user_vector(
        self,
        user_id: str,
        history: Optional[List[str]],
        neg_ids: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], set]:
        """
        Restituisce (user_vector, training_excluded_indices).

        Un utente è considerato "noto" solo se ha almeno una interazione
        in C_pos (training). Essere nell'indice non basta: load_mind_joint()
        inserisce in user2idx anche gli utenti solo-dev, il cui vettore U
        converge a zero per regolarizzazione e non porta segnale utile.

        - Utente con interazioni in training: usa U[u_idx] (già addestrato
          con C_pos e M_neg in fit()) ed esclude gli item già visti.
        - Utente senza interazioni in training (solo-dev o nuovo): fold-in
          sulla history + negativi della sessione corrente.
        """
        if user_id in self.user2idx:
            u_idx = self.user2idx[user_id]
            # check se l'utente ha interazioni reali in training
            has_training_data = (
                self.C_pos is not None and
                self.C_pos.getrow(u_idx).nnz > 0
            )
            if has_training_data:
                user_vec = self.U[u_idx].copy()
                excluded: set = set()
                excluded.update(self.C_pos.getrow(u_idx).indices.tolist())
                return user_vec, excluded

        # user nuovo o solo-dev: fold-in 
        hist = history or []
        if not hist:
            return None, set()
        user_vec = self.compute_user_vector(hist, neg_news_ids=neg_ids)
        excluded = {
            self.news2idx[nid] for nid in hist if nid in self.news2idx
        }
        return user_vec, excluded

    # ------------------------------------------------------------------
    # scoring su pool di candidati
    # ------------------------------------------------------------------

    def _score_candidates(
        self,
        user_vector: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Score finale = (1-w)*norm_s_wals + w*s_pop.
        La normalizzazione min-max è locale ai candidati ricevuti.
        """
        wals_scores = self.V[candidate_indices] @ user_vector

        s_min, s_max = wals_scores.min(), wals_scores.max()
        norm_wals = (
            (wals_scores - s_min) / (s_max - s_min)
            if s_max > s_min
            else np.ones_like(wals_scores) * 0.5
        )

        pop_scores = np.array(
            [self.popularity_dict.get(self.idx2news[idx], 0.0) for idx in candidate_indices],
            dtype=np.float64,
        )
        return (1.0 - self.popularity_weight) * norm_wals + self.popularity_weight * pop_scores

    def _top_wals_candidates(
        self,
        user_vector: np.ndarray,
        all_excluded: set,
        n_candidates: int,
    ) -> np.ndarray:
        """
        Candidate generation per la demo globale: restituisce i
        n_candidates articoli con WALS score più alto (esclusi quelli
        già visti), calcolando il prodotto scalare su tutto V in un
        unico batch.
        """
        wals_scores = self.V @ user_vector 

        if all_excluded:
            excl_arr = np.fromiter(all_excluded, dtype=np.intp, count=len(all_excluded))
            wals_scores[excl_arr] = -np.inf

        k = min(n_candidates, (wals_scores > -np.inf).sum())
        if k == 0:
            return np.array([], dtype=np.intp)

        top_idx = np.argpartition(-wals_scores, k - 1)[:k]
        return top_idx

    # ------------------------------------------------------------------
    # metodo principale
    # ------------------------------------------------------------------

    def recommend_for_user(
        self,
        user_id: str,
        top_k: int = 10,
        history: Optional[List[str]] = None,
        neg_ids: Optional[List[str]] = None,
        seen_items: Optional[List[str]] = None,
        impression_pool: Optional[List[str]] = None,
        n_candidates: int = 500, # pool
    ) -> List[Tuple[str, float]]:
        """
        Genera il ranking per user_id.

        Parametri
        ---------
        user_id        : ID dell'utente (può non essere nell'indice -> fold-in).
        top_k          : numero di raccomandazioni da restituire.
        history        : news lette in precedenza (necessario solo per utenti nuovi).
        seen_items     : news da escludere esplicitamente (sessione corrente).
        impression_pool: se fornito, ranking solo su queste news (evaluation).
        n_candidates   : dimensione del pool intermedio per la demo globale.
        neg_ids        : impression non cliccate della sessione corrente.
        """
        user_vec, train_excluded = self._get_user_vector(user_id, history, neg_ids)

        if user_vec is None:
            return self._popularity_fallback(top_k, seen_items, impression_pool)

        # --- costruzione del pool di candidati ---
        if impression_pool is not None:
            # evaluation: ordiniamo solo le news della sessione corrente.
            # non applichiamo train_excluded: la label del dev è valida
            # anche per articoli già visti in training.
            candidate_indices = np.array(
                [self.news2idx[nid] for nid in impression_pool if nid in self.news2idx],
                dtype=np.intp,
            )
            if candidate_indices.size == 0:
                return []

        else:
            # candidate generation in due fasi.
            #
            # Fase 1 — WALS puro su tutto V: seleziona i
            #   top n_candidates per score di prodotto scalare, escludendo
            #   gli item già visti.
            #
            # Fase 2 — re-ranking con popolarità sui top-n_candidates:
            #   opera su un sottoinsieme già valido, la popolarità
            #   è un segnale complementare.
            extra_excluded = set()
            if seen_items:
                extra_excluded = {
                    self.news2idx[nid] for nid in seen_items if nid in self.news2idx
                }
            all_excluded = train_excluded | extra_excluded

            candidate_indices = self._top_wals_candidates(
                user_vec, all_excluded, n_candidates
            )
            if candidate_indices.size == 0:
                return []

        # --- scoring finale e top-k ---
        final_scores = self._score_candidates(user_vec, candidate_indices)

        k = min(top_k, final_scores.size)
        top_rel = np.argpartition(-final_scores, k - 1)[:k]
        top_rel = top_rel[np.argsort(-final_scores[top_rel])]

        return [
            (self.idx2news[candidate_indices[r]], float(final_scores[r]))
            for r in top_rel
        ]

    # ------------------------------------------------------------------
    # fold-in per utenti nuovi 
    # ------------------------------------------------------------------

    def compute_user_vector(
        self,
        history_news_ids: List[str],
        neg_news_ids: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        x_u = (Y^T C^u Y + lambda*I)^{-1} Y^T C^u p(u)
        A_u = Y^T Y + alpha * Y_pos^T Y_pos + alpha_neg * Y_neg^T Y_neg + lambda*I
        b_u = (1 + alpha) * Y_pos^T 1      

        I negativi comprimono x_u nella direzione degli articoli rifiutati,
        abbassandone lo score senza modificare il target dei positivi.
        """
        f = self.n_factors
        pos_indices = np.array(
            [self.news2idx[nid] for nid in history_news_ids if nid in self.news2idx],
            dtype=np.intp,
        )
        if pos_indices.size == 0:
            return np.zeros(f, dtype=np.float32)

        Y_pos = self.V[pos_indices]
        alpha_vec = self.alpha * np.ones(len(pos_indices), dtype=np.float64)

        YtY = self.V.T @ self.V
        A_u = (YtY
               + Y_pos.T @ (Y_pos * alpha_vec[:, None])
               + self.reg * np.eye(f, dtype=np.float64))
        b_u = Y_pos.T @ (1.0 + alpha_vec)

        # negativi osservati: solo Au, b_u invariato
        if neg_news_ids:
            neg_indices = np.array(
                [self.news2idx[nid] for nid in neg_news_ids if nid in self.news2idx],
                dtype=np.intp,
            )
            if neg_indices.size > 0:
                Y_neg = self.V[neg_indices]
                A_u += self.alpha_neg * (Y_neg.T @ Y_neg)

        try:
            return np.linalg.solve(A_u, b_u).astype(np.float32)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A_u, b_u, rcond=None)[0].astype(np.float32)

    # ------------------------------------------------------------------
    # fallback popolarità
    # ------------------------------------------------------------------

    def _popularity_fallback(
        self,
        top_k: int,
        seen_items: Optional[List[str]],
        impression_pool: Optional[List[str]],
    ) -> List[Tuple[str, float]]:
        pool = impression_pool if impression_pool is not None else list(self.idx2news)
        excluded = set(seen_items or [])
        candidates = [
            (nid, self.popularity_dict.get(nid, 0.0))
            for nid in pool
            if nid not in excluded
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # metadati
    # ------------------------------------------------------------------

    def get_news_metadata(self, news_id: str) -> Dict[str, str]:
        if self.news_df is None:
            return {}
        match = self.news_df[self.news_df["news_id"] == news_id]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {
            "category": str(row.get("category", "")),
            "subcategory": str(row.get("subcategory", "")),
            "title": str(row.get("title", "")),
            "abstract": str(row.get("abstract", "")),
        }