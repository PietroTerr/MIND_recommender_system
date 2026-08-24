"""Interfaccia web Flask per la demo del sistema di raccomandazione.

    python demo_web_flask.py

Apre il browser su http://localhost:5000. Permette di costruire un profilo
utente scegliendo articoli per titolo o caricando un utente reale da MIND,
e restituisce un ranking sull'intero catalogo.
"""

import os
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from data_loader import load_news, load_behaviors
from recommender import NewsRecommenderSystem

# ------------------------------------------------------------
# Configurazione
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
TRAIN_NEWS = ROOT / "data/train/news.tsv"
DEV_NEWS   = ROOT / "data/dev/news.tsv"
TRAIN_BEHAVIORS = ROOT / "data/train/behaviors.tsv"
DEV_BEHAVIORS   = ROOT / "data/dev/behaviors.tsv"

HEARTBEAT_TIMEOUT = 8

ultima_presenza = None

app = Flask(__name__)

# ------------------------------------------------------------
# Caricamento del recommender
# ------------------------------------------------------------
print("Caricamento del recommender...")
if not (ARTIFACTS_DIR / "V.npy").exists():
    sys.exit(f"Modello non trovato in {ARTIFACTS_DIR}. Esegui prima train.py.")

recommender = NewsRecommenderSystem(
    artifacts_dir=str(ARTIFACTS_DIR),
    popularity_csv_path=str(ROOT / "data/train/popularity_data.csv"),
    popularity_weight=0.15,
)

# Metadati completi (train & dev) per la ricerca e le schede
news_train = load_news(str(TRAIN_NEWS))
news_dev   = load_news(str(DEV_NEWS))
all_news   = pd.concat([news_train, news_dev]).drop_duplicates(subset=["news_id"])

titolo     = dict(zip(all_news["news_id"], all_news["title"]))
categoria  = dict(zip(all_news["news_id"], all_news["category"]))
sottocategoria = dict(zip(all_news["news_id"], all_news["subcategory"]))

# Allineamento agli indici del modello
idx2news = recommender.idx2news
news2idx = recommender.news2idx
V = recommender.V
norme = np.linalg.norm(V, axis=1)   # norma di ogni vettore articolo

# Carica i behaviors per recuperare la history degli utenti
train_beh = load_behaviors(str(TRAIN_BEHAVIORS)) if TRAIN_BEHAVIORS.exists() else None
dev_beh   = load_behaviors(str(DEV_BEHAVIORS))   if DEV_BEHAVIORS.exists() else None

# ------------------------------------------------------------
# Funzioni di supporto
# ------------------------------------------------------------
def scheda_articolo(news_id: str, punteggio: float = None) -> dict:
    """Costruisce la scheda di un articolo con metadati e norma."""
    idx = news2idx.get(news_id)
    if idx is None:
        return {
            "id": news_id,
            "titolo": titolo.get(news_id, "(titolo non disponibile)"),
            "categoria": categoria.get(news_id, "?"),
            "sottocategoria": sottocategoria.get(news_id, "?"),
            "norma": 0.0,
            "click": None,
        }
    voce = {
        "id": news_id,
        "titolo": titolo.get(news_id, "(titolo non disponibile)"),
        "categoria": categoria.get(news_id, "?"),
        "sottocategoria": sottocategoria.get(news_id, "?"),
        "norma": round(float(norme[idx]), 3),
        "click": None,   # potremmo caricare da C_pos ma non necessario
    }
    if punteggio is not None:
        voce["punteggio"] = round(float(punteggio), 3)
    return voce

def cerca_articoli(query: str, max_risultati: int = 25) -> list:
    """Restituisce articoli il cui titolo contiene la query, ordinati per norma decrescente."""
    query = query.strip().lower()
    if len(query) < 2:
        return []
    # Scansione lineare (solo 50k articoli, accettabile)
    matches = []
    for idx, nid in enumerate(idx2news):
        if query in titolo.get(nid, "").lower():
            matches.append((idx, norme[idx]))
    # Ordina per norma decrescente
    matches.sort(key=lambda x: -x[1])
    # Restituisce solo i primi max_risultati
    return [scheda_articolo(str(idx2news[idx])) for idx, _ in matches[:max_risultati]]

def history_utente(user_id: str) -> list:
    """Raccoglie tutti gli ID articoli nella history dell'utente da train e dev."""
    history = set()
    for df in (train_beh, dev_beh):
        if df is not None:
            rows = df[df["user_id"] == user_id]
            for _, r in rows.iterrows():
                history.update(r["history"])
    return list(history)

def profilo_utente(user_id: str) -> dict:
    """Carica il profilo dell'utente dai behaviors."""
    hist = history_utente(user_id)
    if not hist:
        return {"trovato": False}
    # Filtra solo gli articoli con rappresentazione nel modello
    noti = [n for n in hist if n in news2idx]
    return {
        "trovato": True,
        "split": "train" if user_id in recommender.user2idx else "dev",  # approssimativo
        "nel_training": user_id in recommender.user2idx,
        "articoli": [scheda_articolo(n) for n in noti],
        "scartati": len(hist) - len(noti),
    }

# ------------------------------------------------------------
# Template HTML (incorporato)
# ------------------------------------------------------------
HTML = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recommender MIND — demo</title>
<style>
  :root { --bordo:#d8d8d8; --tenue:#666; --acc:#2f6db5; --sfondo:#f6f7f9; }
  * { box-sizing: border-box; }
  body { margin:0; font:16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
         color:#1b1b1b; background:var(--sfondo); }
  header { background:#fff; border-bottom:1px solid var(--bordo); padding:14px 22px; }
  header h1 { margin:0; font-size:19px; font-weight:600; }
  header p { margin:3px 0 0; color:var(--tenue); font-size:13.5px; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:20px; padding:20px; }
  @media (max-width: 950px) { main { grid-template-columns: 1fr; } }
  section { background:#fff; border:1px solid var(--bordo); border-radius:8px; padding:16px; }
  h2 { margin:0 0 12px; font-size:15px; text-transform:uppercase;
       letter-spacing:.05em; color:var(--tenue); font-weight:600; }
  input { width:100%; padding:9px 11px; font-size:15px; border:1px solid var(--bordo);
          border-radius:6px; font-family:inherit; }
  .riga { display:flex; gap:8px; align-items:center; }
  .riga input { flex:1; }
  button { padding:9px 14px; font-size:14px; border:1px solid var(--bordo);
           background:#fff; border-radius:6px; cursor:pointer; font-family:inherit; }
  button:hover { background:#eef2f7; border-color:var(--acc); }
  button.primario { background:var(--acc); color:#fff; border-color:var(--acc); }
  button.primario:hover { background:#25599a; }
  .voce { display:flex; gap:10px; align-items:flex-start; padding:8px 0;
          border-bottom:1px solid #eee; }
  .voce:last-child { border-bottom:none; }
  .voce .testo { flex:1; min-width:0; }
  .titolo { font-size:14.5px; }
  .etichette { font-size:12px; color:var(--tenue); margin-top:2px; }
  .cat { background:#eef2f7; border-radius:3px; padding:1px 6px; margin-right:5px; }
  .rango { font-variant-numeric:tabular-nums; font-weight:600; color:var(--acc);
           min-width:2.2em; text-align:right; }
  .punti { font-variant-numeric:tabular-nums; color:var(--tenue); font-size:13px; }
  .vuoto { color:var(--tenue); font-size:14px; padding:10px 0; }
  .nota { font-size:13px; color:var(--tenue); margin-top:12px; }
  .errore { color:#b3261e; font-size:14px; }
  .badge { display:inline-block; font-size:12px; padding:2px 8px; border-radius:10px;
           background:#e7f0fb; color:var(--acc); }
  .badge.attenzione { background:#fdf0e3; color:#8a5300; }
  .scorri { max-height:340px; overflow-y:auto; }
</style>
</head>
<body>
<header>
  <h1>Sistema di raccomandazione — Weighted Matrix Factorisation su MIND</h1>
  <p>Si compone un utente come insieme di documenti graditi; il sistema restituisce
     un ranking sull'intero catalogo.</p>
</header>

<main>
  <section>
    <h2>1 · Costruisci il profilo</h2>
    <div class="riga">
      <input id="ricerca" placeholder="cerca un articolo per titolo (es. NFL, Trump, recipe)" autofocus>
    </div>
    <div id="risultati" class="scorri"></div>

    <div class="riga" style="margin-top:14px">
      <input id="utente" placeholder="oppure carica un utente del dataset (es. U19351)">
      <button onclick="caricaUtente()">Carica</button>
    </div>
    <div id="statoUtente"></div>
  </section>

  <section>
    <h2>2 · Documenti graditi <span id="conta" class="badge">0</span></h2>
    <div id="profilo" class="scorri"><div class="vuoto">Nessun articolo scelto.</div></div>
    <div class="riga" style="margin-top:14px">
      <button class="primario" onclick="raccomanda()">Raccomanda</button>
      <button onclick="svuota()">Svuota</button>
      <span class="nota" style="margin:0">top
        <input id="quanti" type="number" value="10" min="1" max="50"
               style="width:64px;display:inline-block;padding:5px"></span>
    </div>
  </section>

  <section style="grid-column:1 / -1">
    <h2>3 · Ranking dei documenti</h2>
    <div id="ranking"><div class="vuoto">In attesa di un profilo.</div></div>
    <div id="tempi" class="nota"></div>
  </section>
</main>

<script>
let profilo = [];   
let risultatiRicerca = [];

function inviaPresenza() {
  fetch('/api/presenza', {method: 'POST', keepalive: true}).catch(() => {});
}

inviaPresenza();
const timerPresenza = setInterval(inviaPresenza, 2000);
window.addEventListener('pagehide', () => {
  clearInterval(timerPresenza);
  navigator.sendBeacon('/api/presenza');
});

const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function etichette(v) {
  return `<span class="cat">${esc(v.categoria)}</span>
          <span class="cat">${esc(v.sottocategoria)}</span> ${esc(v.id)}`;
}

function disegnaProfilo() {
  document.getElementById('conta').textContent = profilo.length;
  const box = document.getElementById('profilo');
  if (!profilo.length) { box.innerHTML = '<div class="vuoto">Nessun articolo scelto.</div>'; return; }
  box.innerHTML = profilo.map((v, i) => `
    <div class="voce">
      <div class="testo">
        <div class="titolo">${esc(v.titolo)}</div>
        <div class="etichette">${etichette(v)}</div>
      </div>
      <button onclick="togli(${i})">togli</button>
    </div>`).join('');
}

function aggiungi(newsId) {
  const v = risultatiRicerca.find(risultato => risultato.id === newsId);
  if (!v) return;
  if (!profilo.some(p => p.id === v.id)) { profilo.push(v); disegnaProfilo(); }
}
function togli(i) { profilo.splice(i, 1); disegnaProfilo(); }
function svuota() {
  profilo = []; disegnaProfilo();
  document.getElementById('ranking').innerHTML = '<div class="vuoto">In attesa di un profilo.</div>';
  document.getElementById('tempi').textContent = '';
  document.getElementById('statoUtente').innerHTML = '';
}

let attesa;
document.getElementById('ricerca').addEventListener('input', e => {
  clearTimeout(attesa);
  const q = e.target.value;
  attesa = setTimeout(async () => {
    const box = document.getElementById('risultati');
    if (q.trim().length < 2) { box.innerHTML = ''; return; }
    const r = await fetch('/api/cerca?q=' + encodeURIComponent(q));
    const voci = await r.json();
    risultatiRicerca = voci;
    box.innerHTML = voci.length ? voci.map(v => `
      <div class="voce">
        <div class="testo">
          <div class="titolo">${esc(v.titolo)}</div>
          <div class="etichette">${etichette(v)}</div>
        </div>
        <button onclick='aggiungi(${JSON.stringify(v.id)})'>aggiungi</button>
      </div>`).join('') : '<div class="vuoto">Nessun articolo trovato.</div>';
  }, 160);
});

async function caricaUtente() {
  const id = document.getElementById('utente').value.trim();
  const stato = document.getElementById('statoUtente');
  if (!id) return;
  const r = await fetch('/api/utente?id=' + encodeURIComponent(id));
  const d = await r.json();
  if (!d.trovato) { stato.innerHTML = '<div class="errore">Utente non trovato.</div>'; return; }
  profilo = d.articoli;
  disegnaProfilo();
  const badge = d.nel_training
    ? '<span class="badge">presente nel training</span>'
    : '<span class="badge attenzione">NON presente nel training</span>';
  stato.innerHTML = `<div class="nota">${badge}
    profilo letto da split ${esc(d.split)} — ${d.articoli.length} articoli
    utilizzabili${d.scartati ? ', ' + d.scartati + ' senza rappresentazione' : ''}.</div>`;
}

async function raccomanda() {
  const box = document.getElementById('ranking');
  if (!profilo.length) { box.innerHTML = '<div class="errore">Scegli almeno un articolo.</div>'; return; }
  box.innerHTML = '<div class="vuoto">Calcolo…</div>';
  const r = await fetch('/api/raccomanda', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({news: profilo.map(v => v.id),
                          quanti: +document.getElementById('quanti').value})
  });
  const d = await r.json();
  if (d.errore) { box.innerHTML = `<div class="errore">${esc(d.errore)}</div>`; return; }
  box.innerHTML = d.ranking.map((v, i) => `
    <div class="voce">
      <div class="rango">${i + 1}.</div>
      <div class="punti">${v.punteggio.toFixed(3)}</div>
      <div class="testo">
        <div class="titolo">${esc(v.titolo)}</div>
        <div class="etichette">${etichette(v)}</div>
      </div>
    </div>`).join('');
  document.getElementById('tempi').textContent =
    `fold-in e punteggio su ${d.catalogo.toLocaleString('it')} articoli in ${d.millisecondi} ms` +
    ` — esclusi i ${d.profilo.length} gia' nel profilo`;
}
</script>
</body>
</html>
"""

# ------------------------------------------------------------
# Route Flask
# ------------------------------------------------------------
@app.route('/')
def home():
	"""Pagina principale."""
	return render_template_string(HTML)

@app.route('/api/presenza', methods=['POST'])
def api_presenza():
    global ultima_presenza
    ultima_presenza = time.monotonic()
    return '', 204

@app.route('/api/cerca')
def api_cerca():
    query = request.args.get('q', '')
    return jsonify(cerca_articoli(query))

@app.route('/api/utente')
def api_utente():
    user_id = request.args.get('id', '').strip()
    if not user_id:
        return jsonify({"trovato": False})
    return jsonify(profilo_utente(user_id))

@app.route('/api/raccomanda', methods=['POST'])
def api_raccomanda():
    data = request.get_json() or {}
    news_ids = data.get('news', [])
    quanti = max(1, min(50, int(data.get('quanti', 10))))
    if not news_ids:
        return jsonify({"errore": "Nessun articolo fornito."})
    # Verifica che almeno un articolo sia noto nel modello
    noti = [nid for nid in news_ids if nid in recommender.news2idx]
    if not noti:
        return jsonify({"errore": "Nessuno degli articoli scelti compare nel training: "
                                  "senza almeno un articolo noto non è possibile "
                                  "collocare l'utente nello spazio latente."})
    # Usa il recommender per ottenere il ranking
    start = time.perf_counter()
    recs = recommender.recommend_for_user(
        user_id="manual",
        top_k=quanti,
        history=noti,                # questi sono i graditi
        seen_items=None,             # non escludiamo nulla
        impression_pool=None,        # ranking globale
        n_candidates=500,
    )
    elapsed = (time.perf_counter() - start) * 1000
    # Costruisce la risposta con le schede arricchite di punteggio
    ranking = []
    for nid, score in recs:
        scheda = scheda_articolo(nid, score)
        ranking.append(scheda)
    # Profilo (articoli scelti) e scartati (non nel modello)
    profilo_schede = [scheda_articolo(nid) for nid in noti]
    ignorati = [nid for nid in news_ids if nid not in recommender.news2idx]
    return jsonify({
        "ranking": ranking,
        "profilo": profilo_schede,
        "ignorati": ignorati,
        "millisecondi": round(elapsed, 2),
        "catalogo": len(recommender.idx2news),
    })

# ------------------------------------------------------------
# Avvio
# ------------------------------------------------------------
if __name__ == '__main__':
    import webbrowser
    
    print(f"Pronto: {len(recommender.idx2news):,} articoli, k={recommender.n_factors}")

    def arresta_se_senza_browser():
        while True:
            time.sleep(2)
            if (ultima_presenza is not None
                    and time.monotonic() - ultima_presenza > HEARTBEAT_TIMEOUT):
                print("Browser chiuso: arresto del server.", flush=True)
                os._exit(0)

    threading.Thread(target=arresta_se_senza_browser, daemon=True).start()
    
    # Apri il browser dopo un breve ritardo per dare tempo al server di avviarsi
    def apri_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5000')
    
    threading.Thread(target=apri_browser, daemon=True).start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)