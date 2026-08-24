# =====================================================================
# MAIN SCRIPT: Pipeline di Raccomandazione e Valutazione WALS
# =====================================================================

import os
import random
import numpy as np

from data_loader import load_behaviors
from recommender import NewsRecommenderSystem
from metrics import precision_at_k, recall_at_k, mean_average_precision, mean_ndcg_at_k



def main():
    # --- 1. config ---
    artifacts_dir = "artifacts"
    popularity_path = "data/train/popularity_data.csv"
    val_behaviors_path = "data/dev/behaviors.tsv"

    print("=== AVVIO SISTEMA DI RACCOMANDAZIONE WALS ===\n")

    # --- 2. inizializzazione ---
    recommender = NewsRecommenderSystem(
        artifacts_dir=artifacts_dir,
        popularity_csv_path=popularity_path,
        popularity_weight=0.05,
    )

    # --- 3. raccomandazione globale per un utente ---
    # Carichiamo behaviors di train e dev per recuperare la history dell'utente.
    # Per utenti noti (con interazioni in training) la history viene da train.
    # Per utenti solo-dev la history viene dal dev behaviors.tsv.
    train_beh = load_behaviors("data/train/behaviors.tsv")
    dev_beh = load_behaviors(val_behaviors_path)

    user_input = input("\nInserisci l'ID utente (o Enter per il primo): ").strip()
    sample_user_id = (
        user_input if user_input and user_input in recommender.user2idx
        else list(recommender.user2idx.keys())[0]
    )

    # raccoglie history e negativi per la demo
    def get_user_history_and_negs(uid: str):
        history_ids = set()
        neg_ids_set = set()
        for beh_df in [train_beh, dev_beh]:
            rows = beh_df[beh_df["user_id"] == uid]
            for _, r in rows.iterrows():
                history_ids.update(r["history"])
                for imp in r["impressions"]:
                    if "-" not in imp:
                        continue
                    nid, label = imp.rsplit("-", 1)
                    if label == "0":
                        neg_ids_set.add(nid)
        return list(history_ids), list(neg_ids_set - history_ids)

    demo_history, demo_negs = get_user_history_and_negs(sample_user_id)

    # verifica se utente noto (con interazioni in training) o solo-dev
    u_in_train = (
        sample_user_id in recommender.user2idx and
        recommender.C_pos is not None and
        recommender.C_pos.getrow(recommender.user2idx[sample_user_id]).nnz > 0
    )
    if u_in_train:
        print(f"\nRaccomandazioni globali per: {sample_user_id} (utente noto, vettore da training)")
    else:
        print(f"\nRaccomandazioni globali per: {sample_user_id} "
              f"(utente solo-dev, fold-in su {len(demo_history)} articoli history, "
              f"{len(demo_negs)} negativi)")
    print("(ranking su tutto il corpus, articoli già visti esclusi)\n")

    recs = recommender.recommend_for_user(
        user_id = sample_user_id,
        top_k = 10,
        history = demo_history,
        neg_ids = demo_negs,
    )
    for rank, (news_id, score) in enumerate(recs, start=1):
        meta = recommender.get_news_metadata(news_id)
        title = meta.get("title", "N/D")
        cat = meta.get("category", "N/D")
        print(f"{rank:2d}. [{cat}] {title}  (score={score:.4f})  [ID: {news_id}]")

    user_vec = recommender.compute_user_vector(demo_history, neg_news_ids=demo_negs)
    all_scores = recommender.V @ user_vec
    top10 = np.argsort(-all_scores)[:10]

    print("\nTop-10 vicini al vettore fold-in dell'utente:")
    for idx in top10:
        nid = recommender.idx2news[idx]
        meta = recommender.get_news_metadata(nid)
        in_hist = "[v]" if nid in set(demo_history) else ""
        print(f"  [{meta.get('category', 'N/D')}] {meta.get('title', 'N/D')[:60]}  {in_hist}")

    
    # --- 4. evaluation ---
    #
    # le metriche si calcolano sul pool di impression di ogni sessione,
    # non sul corpus intero. Per ogni riga del dev behaviors.tsv:
    #   - pool = tutte le news mostrate in quella sessione (label 0 e 1)
    #   - relevant = le news nel pool con label "1" (cliccate)
    print("\n=== EVALUATION ===")
    random.seed(42)
    if not os.path.exists(val_behaviors_path):
        print(f"[Info] behaviors.tsv non trovato in {val_behaviors_path}.")
        return

    val_df = load_behaviors(val_behaviors_path)

    ranked_lists = []
    pop_ranked_lists = []
    rand_ranked_lists = []
    relevant_lists = []
    
    evaluated = 0
    max_eval = 2000  # impostare a None per valutare l'intero dev split

    for _, row in val_df.iterrows():
        u_id = row["user_id"]
        user_history = row["history"] 
        impressions = row["impressions"]  

        # 1. separa pool, negativi e rilevanti dalle impression
        pool_ids = []
        neg_ids = []   # mostrate e non cliccate: segnale negativo per fold-in
        relevant_ids = []

        for imp in impressions:
            if "-" not in imp:
                continue
            nid, label = imp.rsplit("-", 1)
            pool_ids.append(nid)
            if label == "1":
                relevant_ids.append(nid)
            else:
                neg_ids.append(nid)

        # sessioni non valutabili (nessun click o pool vuoto)
        if not relevant_ids or not pool_ids:
            continue

        # 2. ranking WALS sul pool
        recs = recommender.recommend_for_user(
            user_id=u_id,
            top_k=len(pool_ids),
            history=user_history,
            neg_ids=neg_ids,
            impression_pool=pool_ids,
        )
        if not recs:
            continue

        # salva predizioni WALS e ground truth
        ranked_lists.append([nid for nid, _ in recs])
        relevant_lists.append(relevant_ids)

        # 3. baseline popolarità: ordina pool_ids per click count decrescente
        pool_sorted_by_pop = sorted(
            pool_ids,
            key=lambda nid: recommender.popularity_dict.get(nid, 0.0),
            reverse=True,
        )
        pop_ranked_lists.append(pool_sorted_by_pop)

        # 4. baseline Random: ordine casuale del pool
        pool_shuffled = pool_ids.copy()
        random.shuffle(pool_shuffled)
        rand_ranked_lists.append(pool_shuffled)

        evaluated += 1
        if max_eval is not None and evaluated >= max_eval:
            break

    if not ranked_lists:
        print("[Warning] Nessuna predizione valida.")
        return

    # --- metriche WALS ---
    map_score = mean_average_precision(ranked_lists, relevant_lists)
    ndcg5 = mean_ndcg_at_k(ranked_lists, relevant_lists, k=5)
    ndcg10 = mean_ndcg_at_k(ranked_lists, relevant_lists, k=10)
    p5 = np.mean([precision_at_k(r, rel, 5)  for r, rel in zip(ranked_lists, relevant_lists)])
    p10 = np.mean([precision_at_k(r, rel, 10) for r, rel in zip(ranked_lists, relevant_lists)])
    r10 = np.mean([recall_at_k(r, rel, 10)    for r, rel in zip(ranked_lists, relevant_lists)])

    # --- metriche Random ---
    rand_ndcg5 = mean_ndcg_at_k(rand_ranked_lists, relevant_lists, k=5)
    rand_ndcg10 = mean_ndcg_at_k(rand_ranked_lists, relevant_lists, k=10)
    rand_map = mean_average_precision(rand_ranked_lists, relevant_lists)
    rand_p5 = np.mean([precision_at_k(r, rel, 5)  for r, rel in zip(rand_ranked_lists, relevant_lists)])
    rand_p10 = np.mean([precision_at_k(r, rel, 10) for r, rel in zip(rand_ranked_lists, relevant_lists)])
    rand_r10 = np.mean([recall_at_k(r, rel, 10)    for r, rel in zip(rand_ranked_lists, relevant_lists)])

    # --- metriche Popolarità ---
    pop_ndcg5 = mean_ndcg_at_k(pop_ranked_lists, relevant_lists, k=5)
    pop_ndcg10 = mean_ndcg_at_k(pop_ranked_lists, relevant_lists, k=10)
    pop_map = mean_average_precision(pop_ranked_lists, relevant_lists)
    pop_p5 = np.mean([precision_at_k(r, rel, 5)  for r, rel in zip(pop_ranked_lists, relevant_lists)])
    pop_p10 = np.mean([precision_at_k(r, rel, 10) for r, rel in zip(pop_ranked_lists, relevant_lists)])
    pop_r10 = np.mean([recall_at_k(r, rel, 10)    for r, rel in zip(pop_ranked_lists, relevant_lists)])

    # --- tabella di confronto ---
    print(f"\n{'Metrica':<15} {'Random':>10} {'Popolarità':>12} {'WALS':>10}")
    print("-" * 50)
    print(f"{'NDCG@5':<15} {rand_ndcg5:>10.4f} {pop_ndcg5:>12.4f} {ndcg5:>10.4f}")
    print(f"{'NDCG@10':<15} {rand_ndcg10:>10.4f} {pop_ndcg10:>12.4f} {ndcg10:>10.4f}")
    print(f"{'MAP':<15} {rand_map:>10.4f} {pop_map:>12.4f} {map_score:>10.4f}")
    print(f"{'Precision@5':<15} {rand_p5:>10.4f} {pop_p5:>12.4f} {p5:>10.4f}")
    print(f"{'Precision@10':<15} {rand_p10:>10.4f} {pop_p10:>12.4f} {p10:>10.4f}")
    print(f"{'Recall@10':<15} {rand_r10:>10.4f} {pop_r10:>12.4f} {r10:>10.4f}")
    print(f"\nUtenti/Sessioni valutate: {len(ranked_lists)}")


if __name__ == "__main__":
    main()