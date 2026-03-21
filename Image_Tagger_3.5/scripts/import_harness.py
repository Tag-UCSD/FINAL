"""Lightweight import harness for critical backend modules.

Run with:
    python -m scripts.import_harness
"""

def main() -> None:
    modules = [
        "backend.science.math.color",
        "backend.science.math.glcm",
        "backend.science.math.fractals",
        "backend.science.math.complexity",
        "backend.science.spatial.depth",
        "backend.science.context.cognitive",
        "backend.science.pipeline",
        "backend.services.auth",
        "backend.services.training_export",
        "backend.api.v1_admin",
        "backend.api.v1_supervision",
        "backend.api.v1_annotation",
        "backend.api.v1_discovery",
    ]
    for name in modules:
        try:
            __import__(name)
            print(f"[OK] {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

if __name__ == "__main__":
    main()