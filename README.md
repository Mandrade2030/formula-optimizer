# Formula Optimizer - Mixture Laboratory V2

Streamlit MVP per ottimizzazione DOE iterativa di formulazioni a miscela con vincolo nativo **somma = 100%**.

## Funzioni principali

- Formula base con variabili: Nome, Base, Min, Max, Passo, Lock
- DOE iniziale con vincolo somma 100%
- Tabella principale unica: ID, Iterazione, Source, componenti, Totale, Score
- Score inseribile anche sulle prove suggerite
- Le nuove prove vengono aggiunte automaticamente alla tabella principale
- Gaussian Process + Bayesian-style acquisition
- Anti-duplicati e diversità tra proposte
- Save/Load JSON
- Import CSV storico
- Export CSV/XLSX
- Dashboard con miglior score e grafico convergenza

## Installazione locale

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy su Streamlit Cloud

1. Carica `app.py` e `requirements.txt` nel repository GitHub.
2. Vai su Streamlit Cloud.
3. Seleziona repository e main file `app.py`.
4. Deploy.

## CSV storico

Il CSV deve contenere una colonna per ogni variabile, per esempio:

```csv
A,B,C,D,E,F,Score
53.28,0.10,0.30,29.13,14.26,2.93,95
```

Colonne opzionali: `ID`, `Iterazione`.
