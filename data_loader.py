import csv
import os
from dataclasses import dataclass
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, load_npz, save_npz

# Il dataset non ha intestazione nativa
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

BEHAVIORS_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


@dataclass
class Mappings:
    """user_id <-> indice riga e news_id <-> indice colonna.
    idx2user/idx2news sono la rappresentazione canonica;
    user2idx/news2idx sono ricostruiti da essi.
    """

    user2idx: dict
    idx2user: np.ndarray
    news2idx: dict
    idx2news: np.ndarray


@dataclass
class MindData:
    """Contenitore per uno split del dataset MIND (train/dev)."""

    news: pd.DataFrame
    behaviors: pd.DataFrame
    mappings: Mappings
    C_pos: csr_matrix
    M_neg: csr_matrix

def download_hf_dataset():
    # repo su Hugging Face
    dataset_name = "huyva/MIND-small"

    # cartella locale 
    output_dir = "./data"

    print(f" Avvio del download per il dataset: '{dataset_name}'...")

    try:
        # MIND-small è composto da file grezzi che
        # load_dataset() fonde erroneamente in un'unica tabella.
        # quindi si scaricano come sono, mantenendo la struttura train/dev del repository.
        local_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=output_dir,
        )

        print("\n Download completato con successo")
        print(f" Completato. Path del dataset: {os.path.abspath(local_path)}")

    except Exception as e:
        print(f"\n Si è verificato un errore durante il download: {e}")

def load_news(path: str) -> pd.DataFrame:
    """Carica news.tsv in un DataFrame (un titolo/abstract per news_id)."""
    return pd.read_table(
        path,
        header=None,
        names=NEWS_COLUMNS,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
    )


def load_behaviors(path: str) -> pd.DataFrame:
    """Carica behaviors.tsv in un DataFrame, con colonne history e impressions come liste di stringhe."""
    df = pd.read_table(
        path,
        header=None,
        names=BEHAVIORS_COLUMNS,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
    )
    df["history"] = df["history"].apply(lambda s: s.split() if s else [])
    df["impressions"] = df["impressions"].apply(lambda s: s.split() if s else [])
    return df


def build_mappings(news_df: pd.DataFrame, behaviors_df: pd.DataFrame) -> Mappings:
    """Costruisce le mappature news_id <-> indice colonna e user_id <-> indice riga."""
    idx2news = news_df["news_id"].to_numpy(dtype=str)
    news2idx = {news_id: idx for idx, news_id in enumerate(idx2news)}

    idx2user = np.array(sorted(behaviors_df["user_id"].unique()), dtype=str)
    user2idx = {user_id: idx for idx, user_id in enumerate(idx2user)}

    return Mappings(user2idx=user2idx, idx2user=idx2user, news2idx=news2idx, idx2news=idx2news)


def _pairs_to_csr(pairs: set, shape: tuple) -> csr_matrix:
    """Costruisce una matrice sparsa CSR a partire da un insieme di coppie (riga, colonna)."""

    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape deve essere una tupla di due interi (n_righe, n_colonne)")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("shape deve contenere due interi positivi")
    if not pairs:
        return csr_matrix(shape, dtype=np.float32)
    rows, cols = zip(*pairs)
    data = np.ones(len(pairs), dtype=np.float32)
    return coo_matrix((data, (rows, cols)), shape=shape, dtype=np.float32).tocsr()


def build_feedback_matrices(
    behaviors_df: pd.DataFrame,
    mappings: Mappings,
    include_history: bool = True,
) -> tuple:
    """Costruisce le due strutture sparse del feedback (utenti x news):

    - C_pos: 1 sulle coppie positive (history + click nelle impression, label "1")
    - M_neg: maschera delle coppie "mostrate e non cliccate" (label "0")

    Le due matrici restano disgiunte: se una coppia (u, i) è positiva in una
    sessione, viene rimossa da M_neg anche se in un'altra sessione è stata
    mostrata senza click.
    """

    positive_pairs = set()
    negative_pairs = set()

    for row in behaviors_df.itertuples(index=False):
        u_idx = mappings.user2idx.get(row.user_id)
        if u_idx is None:
            continue

        if include_history:
            for news_id in row.history:
                n_idx = mappings.news2idx.get(news_id)
                if n_idx is not None:
                    positive_pairs.add((u_idx, n_idx)) # history è sempre positiva (click precedente)

        for impression in row.impressions:
            news_id, _, label = impression.rpartition("-")
            n_idx = mappings.news2idx.get(news_id)
            if n_idx is None:
                continue
            if label == "1":
                positive_pairs.add((u_idx, n_idx)) # news consigliate e cliccate
            elif label == "0":
                negative_pairs.add((u_idx, n_idx)) # news consigliate e non cliccate

    negative_pairs -= positive_pairs # rimuove le coppie positive in un'altra sessione

    shape = (len(mappings.idx2user), len(mappings.idx2news))
    C_pos = _pairs_to_csr(positive_pairs, shape)
    M_neg = _pairs_to_csr(negative_pairs, shape)
    return C_pos, M_neg

def build_mappings_joint(
    train_news_df: pd.DataFrame,
    dev_news_df: pd.DataFrame,
    train_behaviors_df: pd.DataFrame,
    dev_behaviors_df: pd.DataFrame,
) -> Mappings:
    """Costruisce un indice unico train & dev.

    Perché è necessario:
    - V (item) viene indicizzato da news2idx. Se un articolo del dev non è
      in news2idx, il modello non può dargli un punteggio (nessuna riga in V) anche dopo il
      fold-in dell'utente.
    - Costruendo l'indice su train & dev prima del training, ogni articolo ottiene
      una colonna in C_pos (e quindi una riga in V dopo WALS). Gli articoli visti
      solo nel dev avranno C_pos tutta a zero su quella colonna e partiranno dal
      vettore inizializzato a zero, ma il modello potrà comunque dargli un punteggio e
      aggiornarli durante il fold-in.
    - Gli utenti  nuovi del dev non avranno una riga in U, ma verranno gestiti tramite fold-in (compute_user_vector).

    Convenzione degli indici:
    - Gli articoli di train vengono prima (indici 0..n_train_news-1), quelli nuovi
      del dev in coda. Questo garantisce che le sottomatrici esistenti non cambino
      indice se si riusa un indice parziale già calcolato.
    """
    # prima train, poi quelli nuovi del dev
    train_ids = train_news_df["news_id"].tolist()
    dev_ids = dev_news_df["news_id"].tolist()

    seen = set(train_ids)
    dev_only_ids = [nid for nid in dev_ids if nid not in seen]

    all_news_ids = train_ids + dev_only_ids
    idx2news = np.array(all_news_ids, dtype=str)
    news2idx = {nid: idx for idx, nid in enumerate(idx2news)}

    # unione sorted
    all_users = sorted(
        set(train_behaviors_df["user_id"].unique()) |
        set(dev_behaviors_df["user_id"].unique())
    )
    idx2user = np.array(all_users, dtype=str)
    user2idx = {uid: idx for idx, uid in enumerate(idx2user)}

    n_train_news = len(train_ids)
    n_dev_only = len(dev_only_ids)
    n_train_users = train_behaviors_df["user_id"].nunique()
    n_dev_only_users = len(set(dev_behaviors_df["user_id"].unique()) -
                           set(train_behaviors_df["user_id"].unique()))

    print(
        f"[Mappings joint] news train={n_train_news}, news solo-dev={n_dev_only}, "
        f"news totali={len(idx2news)} | "
        f"utenti train={n_train_users}, utenti solo-dev={n_dev_only_users}, "
        f"utenti totali={len(idx2user)}"
    )

    return Mappings(user2idx=user2idx, idx2user=idx2user,
                    news2idx=news2idx, idx2news=idx2news)

def load_mind_split(data_dir: str, include_history: bool = True) -> MindData:
    """Carica uno split ('data/train' o 'data/dev')."""

    news_df = load_news(os.path.join(data_dir, "news.tsv"))
    behaviors_df = load_behaviors(os.path.join(data_dir, "behaviors.tsv"))
    mappings = build_mappings(news_df, behaviors_df)
    C_pos, M_neg = build_feedback_matrices(behaviors_df, mappings, include_history=include_history)

    return MindData(news=news_df, behaviors=behaviors_df, mappings=mappings, C_pos=C_pos, M_neg=M_neg)

def load_mind_joint(
    train_dir: str,
    dev_dir: str,
    include_history: bool = True,
) -> tuple:
    """Carica train e dev con l'indice condiviso .

    Restituisce (train_data, dev_data) dove entrambi i MindData condividono
    lo stesso oggetto Mappings, così le matrici C_pos/M_neg sono allineate
    sullo stesso spazio utenti x news e V avrà una riga per ogni news in
    entrambi gli split.
    """

    train_news = load_news(os.path.join(train_dir, "news.tsv"))
    dev_news = load_news(os.path.join(dev_dir,   "news.tsv"))
    train_beh = load_behaviors(os.path.join(train_dir, "behaviors.tsv"))
    dev_beh = load_behaviors(os.path.join(dev_dir,   "behaviors.tsv"))

    joint_mappings = build_mappings_joint(train_news, dev_news, train_beh, dev_beh)

    train_C_pos, train_M_neg = build_feedback_matrices(
        train_beh, joint_mappings, include_history=include_history
    )
    dev_C_pos, dev_M_neg = build_feedback_matrices(
        dev_beh, joint_mappings, include_history=include_history
    )

    # uniamo i due news_df in un unico corpus per i metadati
    all_news = pd.concat([train_news, dev_news], ignore_index=True).drop_duplicates("news_id")

    train_data = MindData(
        news=all_news,
        behaviors=train_beh,
        mappings=joint_mappings,
        C_pos=train_C_pos,
        M_neg=train_M_neg,
    )
    dev_data = MindData(
        news=all_news,
        behaviors=dev_beh,
        mappings=joint_mappings,
        C_pos=dev_C_pos,
        M_neg=dev_M_neg,
    )
    return train_data, dev_data


# --- serializzazione su disco -------------------------------------------
#
# i mappings vanno salvati e ricaricati insieme alle
# matrici U e V prodotte dal training WALS: senza,
# le righe/colonne di U e V non sono più riconducibili a user_id/news_id
# e le matrici salvate sono inutilizzabili.

def save_index(path: str, mappings: Mappings) -> None:
    """Salva su disco solo l'indice (gli array inversi, da cui i dict si ricostruiscono)."""
    np.savez(path, idx2user=mappings.idx2user, idx2news=mappings.idx2news)


def load_index(path: str) -> Mappings:
    """Ricarica l'indice da disco e ricostruisce le mappature dirette."""

    with np.load(path) as data:
        idx2user = data["idx2user"]
        idx2news = data["idx2news"]

    user2idx = {user_id: idx for idx, user_id in enumerate(idx2user)}
    news2idx = {news_id: idx for idx, news_id in enumerate(idx2news)}
    return Mappings(user2idx=user2idx, idx2user=idx2user, news2idx=news2idx, idx2news=idx2news)


def save_artifacts(
    out_dir: str,
    mappings: Mappings,
    C_pos: csr_matrix = None,
    M_neg: csr_matrix = None,
    U: np.ndarray = None,
    V: np.ndarray = None,
) -> None:
    """Salva indice, matrici di feedback e fattori WALS (U, V) nella stessa cartella."""

    os.makedirs(out_dir, exist_ok=True)
    save_index(os.path.join(out_dir, "index.npz"), mappings)

    if C_pos is not None:
        save_npz(os.path.join(out_dir, "C_pos.npz"), C_pos)
    if M_neg is not None:
        save_npz(os.path.join(out_dir, "M_neg.npz"), M_neg)
    if U is not None:
        np.save(os.path.join(out_dir, "U.npy"), U)
    if V is not None:
        np.save(os.path.join(out_dir, "V.npy"), V)


def load_artifacts(out_dir: str) -> dict:
    """Ricarica indice, matrici di feedback e fattori WALS salvati con save_artifacts."""

    result = {"mappings": load_index(os.path.join(out_dir, "index.npz"))}

    optional_files = {
        "C_pos": ("C_pos.npz", load_npz),
        "M_neg": ("M_neg.npz", load_npz),
        "U": ("U.npy", np.load),
        "V": ("V.npy", np.load),
    }
    for key, (filename, loader) in optional_files.items():
        path = os.path.join(out_dir, filename)
        if os.path.exists(path):
            result[key] = loader(path)

    return result


if __name__ == "__main__":
    download_hf_dataset()
    mind_data = load_mind_split("data/train")

    n_users, n_news = mind_data.C_pos.shape
    n_pos = mind_data.C_pos.nnz
    n_neg = mind_data.M_neg.nnz
    overlap = mind_data.C_pos.multiply(mind_data.M_neg).nnz

    print(f"News caricate: {n_news}")
    print(f"Utenti: {n_users}")
    print(f"Coppie positive (C_pos): {n_pos}")
    print(f"Coppie negative osservate (M_neg): {n_neg}")
    print(f"Sovrapposizione C_pos/M_neg: {overlap}")

    save_artifacts("artifacts/train", mind_data.mappings, mind_data.C_pos, mind_data.M_neg)
    reloaded = load_artifacts("artifacts/train")
    assert reloaded["mappings"].idx2user.tolist() == mind_data.mappings.idx2user.tolist()
    assert (reloaded["C_pos"] != mind_data.C_pos).nnz == 0
    assert (reloaded["M_neg"] != mind_data.M_neg).nnz == 0
    print("Round-trip save/load artifacts: OK")
