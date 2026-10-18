# Quickstart

From zero to interrogating your own model in ten minutes.

## 1. Install

```
pip install causallens
```

For torch internals, activation patching, circuit discovery, and the
Crosscoder, add the extra:

```
pip install causallens[torch]
```

## 2. Load your model

Any object with `predict` or `predict_proba` works. The driver detection
picks the right interrogation path automatically.

```python
import joblib
import pandas as pd
from causallens.drivers import detect_driver

model = joblib.load("my_model.pkl")
background = pd.read_csv("my_background.csv")
driver = detect_driver(model, list(background.columns), background.to_numpy())
```

The background matrix is the value source for interventions and
counterfactuals: rows of the model's training distribution. 500 to 5000
rows is the practical range.

Driver kinds supported: sklearn estimators, tree ensembles (LightGBM,
XGBoost, RandomForest), torch modules, ONNX graphs, sklearn pipelines,
ensembles of the above, time series models, backtest engines, decision
logs, and text classifiers.

## 3. Open a session

```python
from causallens.core import MicSession

session = MicSession(driver)
```

Optional: attach a concept pack to ground features in domain language.

```python
from causallens.core import ConceptPack

pack = ConceptPack.load("concepts/finance.json")
session = MicSession(driver, pack)
```

## 4. Ask the first questions

Attribution, the what:

```python
session.get_attribution(0)
```

Intervention, the what-if:

```python
session.intervene(0, {"income": 95000})
```

Counterfactual, the how-to-flip:

```python
session.run_counterfactual(0)
session.diverse_counterfactuals(0, n_paths=3)
```

Faithfulness, the can-I-trust-it:

```python
session.run_faithfulness()
```

The full deterministic report, no LLM needed:

```python
print(session.offline_report(sample_index=0))
```

## 5. Go deeper by level

| Level | Question | Tool |
|---|---|---|
| 1 | Which features matter | `get_attribution`, `fit_fastshap` |
| 2 | How do features relate | `get_causal_graph(discovery="pc")` |
| 3 | Direct or mediated | `mediation_decomposition` |
| 4 | Which units compute it | `discover_circuit`, `neuron_ranking` |
| 5 | Which parameters matter | `param_ablation` |
| 6 | What do the concepts mean | `sae_features`, `crosscoder_features` |

Fairness, leakage, drift, backtest audits, and the LLM investigator loop
are covered in [Tools](tools.md).

## 6. Keep the evidence

```python
from causallens.ledger import Ledger

ledger = Ledger("my_ledger.db")
session.attach_ledger(ledger, model_version="v1")
```

Every result is then recorded with a chain hash; any silent edit breaks
the chain and `session.case_briefing()` says where.

## 7. Command line

```
python -m causallens report --model model.pkl --background bg.csv
python -m causallens audit --model model.pkl --background bg.csv
python -m causallens hero
python -m causallens serve --model model.pkl --background bg.csv
```

`serve` starts an MCP server so any MCP client, including an LLM
workbench, can run the interrogation tools.
