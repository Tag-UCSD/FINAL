#!/bin/bash
echo "🚀 Starting Image Tagger v3.1..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop."
    exit 1
fi

# Governance check (Guardian)
echo "🔒 Running Guardian (governance) checks..."
if command -v python3 &> /dev/null; then
    if [ -f "governance.lock" ]; then
        python3 scripts/guardian.py verify
        GUARDIAN_RC=$?
        if [ "$GUARDIAN_RC" -ne 0 ]; then
            echo "❌ Guardian verification failed (rc=$GUARDIAN_RC). Aborting install."
            exit $GUARDIAN_RC
        fi
    else
        echo "⚠️ governance.lock not found; creating initial baseline with 'guardian.py freeze'."
        python3 scripts/guardian.py freeze
        if [ "$?" -ne 0 ]; then
            echo "❌ Guardian freeze failed. Aborting install."
            exit 1
        fi
    fi
else
    echo "⚠️ python3 not found; skipping Guardian checks."
fi




echo "🐳 Building Containers..."
cd deploy
docker-compose up -d --build

echo "🌱 Seeding Database..."
sleep 5
docker-compose exec -T api python3 backend/scripts/seed_tool_configs.py
docker-compose exec -T api python3 backend/scripts/seed_attributes.py

echo "✅ SYSTEM ONLINE at http://localhost:8080"


echo "🧪 Running smoke tests (API + Science)..."
docker-compose exec -T api python3 scripts/smoke_api.py
docker-compose exec -T api python3 scripts/smoke_science.py
echo "✅ Smoke tests passed."
echo "🧪 Running pytest API smoketests..."
docker-compose exec -T api pytest -q tests/test_v3_api.py
if [ "$?" -ne 0 ]; then
    echo "❌ Pytest API smoketests failed."
    exit 1
fi
echo "✅ Pytest API smoketests passed."