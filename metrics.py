# METRICHE DI VALUTAZIONE DEL MODELLO.

import numpy as np

def precision_at_k(ranked_items, relevant_items, k: int) -> float:
    """Frazione di item rilevanti fra i primi k della lista ordinata."""
    if k <= 0:
        return 0.0
    relevant_items = set(relevant_items)
    top_k = ranked_items[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / len(top_k)


def recall_at_k(ranked_items, relevant_items, k: int) -> float:
    """Frazione di item rilevanti totali recuperati entro i primi k."""
    relevant_items = set(relevant_items)
    if not relevant_items:
        return 0.0
    top_k = ranked_items[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / len(relevant_items)


def r_precision(ranked_items, relevant_items) -> float:
    """Precision calcolata al rango R = numero di item rilevanti (precision_at_k con k=R)."""
    relevant_items = set(relevant_items)
    r = len(relevant_items)
    if r == 0:
        return 0.0
    return precision_at_k(ranked_items, relevant_items, r)


# --- curva precision-recall --------------------------------------------------

def precision_recall_points(ranked_items, relevant_items):
    """Calcola (recall, precision) ad ogni rango in cui si incontra un item rilevante (curva PR non interpolata)."""
    relevant_items = set(relevant_items)
    n_relevant = len(relevant_items)
    if n_relevant == 0:
        return np.array([]), np.array([])

    hits = 0
    recalls, precisions = [], []
    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant_items:
            hits += 1
            recalls.append(hits / n_relevant)
            precisions.append(hits / rank)

    return np.asarray(recalls), np.asarray(precisions)


def interpolated_precision(recalls: np.ndarray, precisions: np.ndarray, recall_levels: np.ndarray) -> np.ndarray:
    """Interpolazione."""
    interp = np.zeros(len(recall_levels), dtype=np.float64)
    for idx, level in enumerate(recall_levels):
        mask = recalls >= level
        interp[idx] = precisions[mask].max() if mask.any() else 0.0
    return interp


def eleven_point_precision(ranked_items, relevant_items) -> np.ndarray:
    """Precision interpolata agli 11 livelli di recall standard {0.0, 0.1, ..., 1.0}."""
    recall_levels = np.linspace(0.0, 1.0, 11)
    recalls, precisions = precision_recall_points(ranked_items, relevant_items)
    if len(recalls) == 0:
        return np.zeros(11)
    return interpolated_precision(recalls, precisions, recall_levels)


def mean_eleven_point_precision(ranked_items_list, relevant_items_list) -> np.ndarray:
    """Media della curva a 11 punti su più utenti."""
    curves = [
        eleven_point_precision(ranked, relevant)
        for ranked, relevant in zip(ranked_items_list, relevant_items_list)
        if len(relevant) > 0
    ]
    if not curves:
        return np.zeros(11)
    return np.mean(curves, axis=0)


# --- average precision / MAP --------------------------------------------------

def average_precision(ranked_items, relevant_items) -> float:
    """Average Precision (AP) per un singolo utente: media della precision calcolata
    ad ogni rango in cui compare un item rilevante."""
    relevant_items = set(relevant_items)
    n_relevant = len(relevant_items)
    if n_relevant == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant_items:
            hits += 1
            sum_precisions += hits / rank

    return sum_precisions / n_relevant


def mean_average_precision(ranked_items_list, relevant_items_list) -> float:
    """MAP: media dell'Average Precision su tutti gli utenti con almeno un item rilevante."""
    aps = [
        average_precision(ranked, relevant)
        for ranked, relevant in zip(ranked_items_list, relevant_items_list)
        if len(relevant) > 0
    ]
    return float(np.mean(aps)) if aps else 0.0


# --- NDCG (Normalized Discounted Cumulative Gain) ----------------------------
#
# NDCG@k è la metrica standard del benchmark MIND ufficiale.
#
#   - MAP tratta ogni rilevante come binario e considera solo la sua posizione
#     relativa agli altri rilevanti.
#   - NDCG@k penalizza in modo logaritmico la posizione assoluta nel ranking
#     e cattura meglio quindi quanto "in cima" sono i rilevanti,
#     indipendentemente da quanti altri rilevanti ci siano.
 
def dcg_at_k(ranked_items, relevant_items, k: int) -> float:
    """Discounted Cumulative Gain @k con rilevanza binaria."""
    relevant_items = set(relevant_items)
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_items:
            dcg += 1.0 / np.log2(rank + 1)
    return dcg
 
 
def ndcg_at_k(ranked_items, relevant_items, k: int) -> float:
    """Normalized DCG @k con rilevanza binaria.
 
    IDCG è il DCG del ranking ideale: i min(R, k) rilevanti nei primi posti.
    Se non ci sono rilevanti, restituisce 0.0 (sessione non valutabile).
    """
    relevant_items = set(relevant_items)
    n_relevant = len(relevant_items)
    if n_relevant == 0 or k <= 0:
        return 0.0
 
    dcg = dcg_at_k(ranked_items, relevant_items, k)
    # IDCG: somma dei primi min(n_relevant, k) termini del ranking perfetto
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(n_relevant, k) + 1))
 
    return dcg / idcg if idcg > 0 else 0.0
 
 
def mean_ndcg_at_k(ranked_items_list, relevant_items_list, k: int) -> float:
    """Media di NDCG@k su tutti gli utenti con almeno un item rilevante."""
    scores = [
        ndcg_at_k(ranked, relevant, k)
        for ranked, relevant in zip(ranked_items_list, relevant_items_list)
        if len(relevant) > 0
    ]
    return float(np.mean(scores)) if scores else 0.0