# Active Learning Pipeline v3.0

This document describes the workflow for refining AI models using human validation data.

## 1. The Loop
1.  **Inference**: System predicts labels (e.g., "Modern: 0.9") using `gemini-1.5-flash`.
2.  **Correction**: Human tagger in Workbench changes value to "Modern: 0.4".
3.  **Capture**: System flags this Region/Image as `is_training_candidate=True`.
4.  **Export**: Admin exports candidates via `POST /api/training/export`.
5.  **Fine-Tuning**: Data is sent to OpenAI/Google fine-tuning endpoint.
6.  **Deployment**: New model ID is updated in `ToolConfig` via the Admin Panel.

## 2. Export Format (JSONL)
The system exports data in standard JSONL format for fine-tuning:
`{"messages": [{"role": "user", "content": [IMAGE]}, {"role": "assistant", "content": "Label"}]}`

## 3. Triggering a Training Run
Use the Admin API or CLI:
`curl -X POST http://localhost:8000/api/v1/admin/training/export -d '{"min_quality": "high"}'`