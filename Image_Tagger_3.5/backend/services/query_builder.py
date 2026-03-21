from typing import Any, Dict, List, Optional, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_

from backend.models.image import Image
from backend.models.attribute import Attribute
from backend.models.annotation import Validation


class QueryBuilder:
    """
    Query builder for Explorer search.

    Supported filters (keys in filters dict):
      - "text": str (simple substring match on Image.filename/storage_path/meta_data json)
      - "attribute_ids": List[int] (images validated positive for any of these attributes)
      - "min_score": float (minimum validation score threshold)
      - "has_meta": bool (images with non-null meta_data)
      - "created_after": datetime/date ISO string

    The primary public method is execute(filters, page, page_size).
    search_images(query) remains as a backward-compatible adapter.
    """

    def __init__(self, db: Session, user_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id

    def execute(
        self,
        filters: Dict[str, Any],
        page: int = 1,
        page_size: int = 24,
    ) -> Dict[str, Any]:
        q = self.db.query(Image)

        clauses = []

        text = filters.get("text")
        if text:
            like = f"%{text}%"
            clauses.append(
                or_(
                    Image.filename.ilike(like),
                    Image.storage_path.ilike(like),
                    Image.meta_data.cast(String).ilike(like),
                )
            )

        attribute_ids = filters.get("attribute_ids") or []
        min_score = filters.get("min_score", 0.5)
        if attribute_ids:
            q = q.join(Validation).filter(
                Validation.attribute_id.in_(attribute_ids),
                Validation.value >= min_score,
            )

        if filters.get("has_meta") is True:
            clauses.append(Image.meta_data.isnot(None))

        if clauses:
            q = q.filter(and_(*clauses))

        total = q.count()
        items = (
            q.order_by(Image.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        return {"items": items, "total": total}

    # ------------------------------------------------------------------
    # Backward-compatibility adapter for older router/frontend calls.
    # ------------------------------------------------------------------
    def search_images(self, query: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Accepts either:
          - dict with keys {filters, page, page_size}
          - SearchQuery pydantic object with those fields.
        Delegates to execute().
        """
        if isinstance(query, dict):
            filters = query.get("filters", {}) or {}
            page = query.get("page", 1) or 1
            page_size = query.get("page_size", 24) or 24
        else:
            filters = getattr(query, "filters", {}) or {}
            page = getattr(query, "page", 1) or 1
            page_size = getattr(query, "page_size", 24) or 24
        return self.execute(filters=filters, page=page, page_size=page_size)