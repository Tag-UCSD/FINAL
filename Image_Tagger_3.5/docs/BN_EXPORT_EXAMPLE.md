# BN Export Example

This note shows how to call the BN export endpoint from Python and turn
the result into a pandas DataFrame for further modeling.

```python
import requests
import pandas as pd

API_URL = "http://localhost:8000/api/v1/export/bn-snapshot"

# If your API enforces roles via headers, include them as needed.
headers = {
    "X-User-Role": "admin",
}

resp = requests.get(API_URL, headers=headers)
resp.raise_for_status()

rows = resp.json()
df = pd.DataFrame(rows)

print(df.head())

# From here you can feed `df` into your Bayesian network tooling of choice,
# or export it to CSV:
df.to_csv("bn_snapshot.csv", index=False)
```


## Including VLM cognitive and affective attributes

If your pipeline has the VLM analyzer enabled, the BN snapshot rows will
include continuous attributes such as:

- `cognitive.coherence`, `cognitive.complexity`, `cognitive.legibility`,
  `cognitive.mystery`, `cognitive.restoration`
- `affect.cozy`, `affect.welcoming`, `affect.tranquil`,
  `affect.scary`, `affect.jarring`

You can discretize these into 3-state BN nodes as follows:

```python
def bin_3(x: float) -> str:
    if x < 0.33:
        return "LOW"
    if x < 0.66:
        return "MID"
    return "HIGH"

mapping = {
    "cognitive.coherence": "COHERENCE",
    "cognitive.complexity": "COMPLEXITY",
    "cognitive.legibility": "LEGIBILITY",
    "cognitive.mystery": "MYSTERY",
    "cognitive.restoration": "RESTORATION",
    "affect.cozy": "AFFECT_COSY",
    "affect.welcoming": "AFFECT_WELCOMING",
    "affect.tranquil": "AFFECT_TRANQUIL",
    "affect.scary": "AFFECT_SCARY",
    "affect.jarring": "AFFECT_JARRING",
}

for col, node in mapping.items():
    if col in df.columns:
        df[node] = df[col].apply(bin_3)

df.to_csv("bn_snapshot_with_bins.csv", index=False)
```

See `docs/BN_MAPPING_COG_AFFECT.md` and
`docs/BN_PRIORS_COG_AFFECT_EXAMPLE.csv` for more details.
