# Fattorizzazione feedback matrix tramite WALS (Weighted Alternating Least Squares) seguendo il modello di Hu per l'implicit feedback. 
# Il modello è descritto in "Collaborative Filtering for Implicit Feedback Datasets" di Hu et al. (2008).
# La matrice di feedback è rappresentata come una matrice sparsa di interazioni utente-articolo, 
# dove le interazioni positive sono pesate più fortemente rispetto a quelle negative.

import numpy as np
import scipy.sparse as sp


class ImplicitWALS:
    """
    Weighted Alternating Least Squares per implicit feedback.
    Esteso a tre livelli di confidenza per sfruttare M_neg di MIND.
    """

    def __init__(
        self,
        n_factors: int = 50,    # fattori latenti 
        reg: float = 0.01,  # regulation term
        alpha: float = 40.0,    # peso confidenza impression positiva 
        alpha_neg: float = 20.0,    # peso confidenza impression negativa
        n_iterations: int = 30, # training sweeps 
    ):
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha       
        self.alpha_neg = alpha_neg   
        self.n_iterations = n_iterations

        self.U = None   # (n_users, n_factors)
        self.V = None   # (n_items, n_factors)

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def fit(self, C_pos: sp.csr_matrix, M_neg: sp.csr_matrix = None) -> "ImplicitWALS":
        """
        Addestra il modello WALS.

        C_pos: matrice sparsa (n_users x n_items) delle interazioni positive
        M_neg: matrice sparsa (n_users x n_items) delle impression non cliccate.
        """
        n_users, n_items = C_pos.shape

        rng = np.random.default_rng(42)
        self.U = rng.normal(0, 0.01, (n_users, self.n_factors)).astype(np.float64)
        self.V = rng.normal(0, 0.01, (n_items, self.n_factors)).astype(np.float64)

        C_csr = C_pos.tocsr()
        C_csc = C_pos.tocsc()

        # M_neg in formato CSR per righe-utenti e CSC per colonne-item
        use_neg = M_neg is not None and M_neg.nnz > 0
        if use_neg:
            N_csr = M_neg.tocsr()
            N_csc = M_neg.tocsc()
        else:
            N_csr = N_csc = None

        I_f = np.eye(self.n_factors, dtype=np.float64)

        for iteration in range(self.n_iterations):
            print(f"--- WALS Iterazione {iteration + 1}/{self.n_iterations} ---")

            # ----------------------------------------------------------
            # fase 1: aggiornamento U (item V fissi)
            # ----------------------------------------------------------
            VtV = self.V.T @ self.V   # (f x f), precomputato una volta

            for u in range(n_users):
                # --- indici e rating positivi ---
                pos_idx = C_csr.indices[C_csr.indptr[u]:C_csr.indptr[u+1]]
                pos_data = C_csr.data[C_csr.indptr[u]:C_csr.indptr[u+1]]

                # --- indici negativi osservati ---
                if use_neg:
                    neg_idx = N_csr.indices[N_csr.indptr[u]:N_csr.indptr[u+1]]
                else:
                    neg_idx = np.array([], dtype=np.intp)

                if len(pos_idx) == 0 and len(neg_idx) == 0:
                    self.U[u] = np.zeros(self.n_factors)
                    continue

                # A_u = Y^T Y  +  alpha_pos * Y_pos^T Y_pos
                #               +  alpha_neg * Y_neg^T Y_neg
                #               +  lambda * I
                A_u = VtV.copy()

                if len(pos_idx) > 0:
                    Y_pos = self.V[pos_idx]
                    conf_pos = self.alpha * pos_data    # alpha * r_ui (r_ui=1 -> alpha)
                    A_u += Y_pos.T @ (Y_pos * conf_pos[:, None])
                    # b_u = sum_{pos} (1 + alpha) * y_i
                    c_pos = 1.0 + conf_pos
                    b_u = Y_pos.T @ c_pos
                else:
                    b_u = np.zeros(self.n_factors, dtype=np.float64)

                if len(neg_idx) > 0:
                    Y_neg = self.V[neg_idx]
                    # p_ui=0 -> nessun contributo al termine noto
                    A_u += self.alpha_neg * (Y_neg.T @ Y_neg)

                A_u += self.reg * I_f
                self.U[u] = np.linalg.solve(A_u, b_u)

            # ----------------------------------------------------------
            # fase 2: aggiornamento V (utenti U fissi)
            # ----------------------------------------------------------
            UtU = self.U.T @ self.U   # (f x f), precomputato una volta

            for i in range(n_items):
                # --- indici e rating positivi ---
                pos_idx = C_csc.indices[C_csc.indptr[i]:C_csc.indptr[i+1]]
                pos_data = C_csc.data[C_csc.indptr[i]:C_csc.indptr[i+1]]

                # --- indici negativi osservati ---
                if use_neg:
                    neg_idx = N_csc.indices[N_csc.indptr[i]:N_csc.indptr[i+1]]
                else:
                    neg_idx = np.array([], dtype=np.intp)

                if len(pos_idx) == 0 and len(neg_idx) == 0:
                    self.V[i] = np.zeros(self.n_factors)
                    continue

                A_i = UtU.copy()

                if len(pos_idx) > 0:
                    X_pos = self.U[pos_idx]
                    conf_pos = self.alpha * pos_data
                    A_i += X_pos.T @ (X_pos * conf_pos[:, None])
                    c_pos = 1.0 + conf_pos
                    b_i = X_pos.T @ c_pos
                else:
                    b_i = np.zeros(self.n_factors, dtype=np.float64)

                if len(neg_idx) > 0:
                    X_neg = self.U[neg_idx]
                    A_i += self.alpha_neg * (X_neg.T @ X_neg)

                A_i += self.reg * I_f
                self.V[i] = np.linalg.solve(A_i, b_i)

        return self

    # ------------------------------------------------------------------
    # inferenza
    # ------------------------------------------------------------------

    def predict(self, user_idx: int) -> np.ndarray:
        if self.U is None or self.V is None:
            raise ValueError("Modello non addestrato. Esegui prima fit().")
        return self.V @ self.U[user_idx]

    # top 10 articoli raccomandati 
    def recommend(
        self,
        user_idx: int,
        top_k: int = 10,
        exclude_items: np.ndarray = None,
    ) -> np.ndarray:
        scores = self.predict(user_idx)
        if exclude_items is not None and len(exclude_items) > 0:
            scores = scores.copy()
            scores[exclude_items] = -np.inf
        return np.argsort(-scores)[:top_k]