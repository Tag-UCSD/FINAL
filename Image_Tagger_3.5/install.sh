#!/bin/bash
set -e

VERSION="dev"
[ -f VERSION ] && VERSION=$(cat VERSION)

echo "Starting Image Tagger Explorer (v${VERSION})"

if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker Desktop."
    exit 1
fi

echo "Building containers..."
cd deploy
docker-compose up -d --build

echo "Seeding database..."
sleep 5
docker-compose exec -T api python3 backend/scripts/seed_attributes.py

echo "System online at http://localhost:8080"
