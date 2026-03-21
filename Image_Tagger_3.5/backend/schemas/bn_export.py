"""Pydantic schemas for BN-ready export rows."""

from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel


class BNRow(BaseModel):
    image_id: int
    source: str
    indices: Dict[str, Optional[float]]
    bins: Dict[str, Optional[str]]
    # Optional inter-rater reliability summary for this image.
    # If present, this is typically an average agreement metric across
    # attributes, following the same pairwise-agreement logic used in
    # the Supervisor IRR endpoint.
    agreement_score: Optional[float] = None
    irr_bin: Optional[str] = None

from typing import List, Literal


class BNVariable(BaseModel):
    """Codebook entry describing a single BN variable.

    This is designed to be easily serialised to JSON and consumed by
    downstream BN / probabilistic programming tools.
    """

    name: str
    # Optional human-readable label; when omitted, `name` can be used.
    label: Optional[str] = None
    # Short description of the variable's meaning.
    description: Optional[str] = None
    # Coarse role of the variable in the export.
    # - "index": continuous science index (e.g. fractal_d, complexity)
    # - "bin": ordinal / categorical bin (e.g. low / mid / high)
    # - "meta": derived metadata (e.g. irr_bin)
    # - "id": identifier fields (e.g. image_id)
    role: Literal["index", "bin", "meta", "id"] = "index"
    # Variable type from the perspective of BN tools.
    var_type: Literal["continuous", "discrete", "ordinal"] = "continuous"
    # Optional enumerated states. For continuous variables this is typically None.
    states: Optional[List[str]] = None


class BNCodebook(BaseModel):
    """Container for the BN export codebook.

    The `variables` list fully describes the domain of each column in the
    `/v1/export/bn-snapshot` endpoint so that downstream tools do not need
    to guess at data types or valid states.
    """

    variables: List[BNVariable]

class BNValidationRow(BaseModel):
    """Flattened per-validation row for hierarchical / multi-level models.

    Each row corresponds to a single Validation record, restricted to
    science-pipeline and manual sources. Downstream tools can group by
    `image_id`, `user_id`, or `attribute_key` as needed.
    """

    image_id: int
    user_id: Optional[int]
    attribute_key: str
    value: float
    source: str
    duration_ms: int
