# =====================================================================
# SCRIPT DI TRAINING: Elaborazione dati, WALS fit e Salvataggio Artefatti
# =====================================================================
#
# STRATEGIA INDICE JOINT (train & dev)
# -------------------------------------
# Il modello WALS apprende vettori item (V) solo per gli articoli
# presenti nell'indice al momento del training. Se l'indice copre
# solo il training set, gli articoli nuovi del dev risultano
# "sconosciuti" e non possono essere scorati, nemmeno dopo il fold-in.

import os
import numpy as np
import scipy.sparse as sp

from data_loader import load_mind_joint, save_artifacts
from wals import ImplicitWALS


def main():
    # --- 1. config percorsi ---
    train_dir = r"data/train"
    dev_dir = r"data/dev"
    artifacts_dir = "artifacts"

    os.makedirs(artifacts_dir, exist_ok=True)

    # --- 2. load joint idx ---
    print("=== 1. CARICAMENTO DEI DATI (indice train ∪ dev) ===")
    train_data, dev_data = load_mind_joint(train_dir, dev_dir, include_history=True)

    n_users, n_news = train_data.C_pos.shape
    print(f"Matrice di training: {n_users} utenti × {n_news} news")
    print(f"  C_pos non-zeros: {train_data.C_pos.nnz}")
    print(f"  M_neg non-zeros: {train_data.M_neg.nnz}")

    # --- 3. training WALS ---
    print("\n=== 2. ADDESTRAMENTO MODELLO WALS ===")
    wals_model = ImplicitWALS(
        n_factors=50, # fattori latenti 
        reg=0.01,   # regulation factor
        alpha=40.0, # confidenza per impression negative  
        alpha_neg=20.0, # confidenza per impression negative
        n_iterations=30,
    )
    # no contaminazione: fit() riceve sia C_pos che M_neg solo dal training set.
    wals_model.fit(train_data.C_pos, train_data.M_neg)

    # --- 4. SALVATAGGIO ARTEFATTI ---
    print("\n=== 3. SALVATAGGIO ARTEFATTI ===")
    save_artifacts(
        out_dir=artifacts_dir,
        mappings=train_data.mappings,
        C_pos=train_data.C_pos,
        M_neg=train_data.M_neg,
        U=wals_model.U,
        V=wals_model.V,
    )

    # salva anche il news_df unificato come parquet per i metadati nel recommender
    news_parquet = os.path.join(artifacts_dir, "news.parquet")
    train_data.news.to_parquet(news_parquet, index=False)

    print(f"\nArtefatti salvati in '{artifacts_dir}/':")
    for f in ["index.npz", "C_pos.npz", "M_neg.npz", "U.npy", "V.npy", "news.parquet"]:
        p = os.path.join(artifacts_dir, f)
        size_kb = os.path.getsize(p) // 1024 if os.path.exists(p) else -1
        print(f"  {f:20s}  {size_kb:6d} KB")


if __name__ == "__main__":
    main()