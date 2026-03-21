# Mapping of VLM-derived cognitive & affective attributes to BN nodes

Image Tagger v3.4.x writes VLM-derived attributes into the Validation table
with keys of the form:

- `cognitive.coherence`
- `cognitive.complexity`
- `cognitive.legibility`
- `cognitive.mystery`
- `cognitive.restoration`

and affective judgments:

- `affect.cozy`
- `affect.welcoming`
- `affect.tranquil`
- `affect.scary`
- `affect.jarring`

For Bayesian network modeling we recommend introducing discrete BN nodes with
3-state bins (LOW / MID / HIGH) derived from the continuous scores in `[0, 1]`.

## Suggested BN node names

Cognitive:

- `COHERENCE`
- `COMPLEXITY`
- `LEGIBILITY`
- `MYSTERY`
- `RESTORATION`

Affective:

- `AFFECT_COSY`
- `AFFECT_WELCOMING`
- `AFFECT_TRANQUIL`
- `AFFECT_SCARY`
- `AFFECT_JARRING`

## Suggested binning scheme

For each scalar attribute `x ∈ [0, 1]` (e.g. `cognitive.coherence`):

- `LOW`  if `0.0 ≤ x < 0.33`
- `MID`  if `0.33 ≤ x < 0.66`
- `HIGH` if `0.66 ≤ x ≤ 1.0`

You can materialize these as new columns on the BN snapshot DataFrame, for
example:

```python
def bin_3(x: float) -> str:
    if x < 0.33:
        return "LOW"
    if x < 0.66:
        return "MID"
    return "HIGH"

for key, node in [
    ("cognitive.coherence", "COHERENCE"),
    ("cognitive.complexity", "COMPLEXITY"),
    ("cognitive.legibility", "LEGIBILITY"),
    ("cognitive.mystery", "MYSTERY"),
    ("cognitive.restoration", "RESTORATION"),
    ("affect.cozy", "AFFECT_COSY"),
    ("affect.welcoming", "AFFECT_WELCOMING"),
    ("affect.tranquil", "AFFECT_TRANQUIL"),
    ("affect.scary", "AFFECT_SCARY"),
    ("affect.jarring", "AFFECT_JARRING"),
]:
    if key in df.columns:
        df[node] = df[key].apply(bin_3)
```

This keeps the BN-facing variables compact while preserving the ordinal
structure of the original VLM scores.
