#!/bin/bash
# 🚀 Anti-Gravity Setup for Full Stack VM (Track A)
# Tested on Ubuntu 22.04 LTS
#
# This script is intended for instructors / TAs who want to run the
# full Image Tagger stack on a cloud VM or lab server with minimal
# manual setup.
#
# Usage:
#   1. Copy the Image_Tagger_v3.4.71_anti_gravity_cloud_full.zip artifact
#      to the VM (or clone the repo).
#   2. Run:
#         chmod +x infra/cloud/full_stack_vm_setup.sh
#         ./infra/cloud/full_stack_vm_setup.sh
#
#   3. Follow the printed instructions to access the UI.
#
set -e

echo "[Setup] 🔄 Updating system..."
sudo apt-get update -y
sudo apt-get install -y unzip curl git

# 1. Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[Setup] 🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker "$USER" || true
    echo "[Setup] ⚠️  You may need to log out and back in for Docker permissions to take effect."
else:
    echo "[Setup] ✅ Docker already installed."
fi

# 2. Unpack repo if a ZIP is present in the current directory
if [ -f "Image_Tagger_v3.4.71_anti_gravity_cloud_full.zip" ]; then
    echo "[Setup] 📦 Unzipping artifact..."
    rm -rf image_tagger
    mkdir -p image_tagger
    unzip -o Image_Tagger_v3.4.71_anti_gravity_cloud_full.zip -d image_tagger
    cd image_tagger
else
    echo "[Setup] ℹ️  Zip not found. Assuming we are already in the repo root."
fi

# 3. Run auto installer
if [ -x "./auto_install.sh" ]; then
    echo "[Setup] 🏗️  Running auto_install.sh..."
    ./auto_install.sh
else
    echo "[Setup] ❌ auto_install.sh not found or not executable."
    echo "       Please ensure you are in the Image Tagger repo root."
    exit 1
fi

echo "----------------------------------------------------------------"
echo "✅ FULL STACK SHOULD NOW BE RUNNING (or ready to start via docker-compose)."
echo ""
echo "Typical ports:"
echo "   - Backend API: http://localhost:8000"
echo "   - Frontend UI: http://localhost:8080 (or as defined in docker-compose)"
echo ""
echo "☁️  To expose this to the public internet for demos, you can use ngrok:"
echo "   1. Install:  snap install ngrok   (or follow ngrok's official instructions)"
echo "   2. Run:      ngrok http 8080"
echo ""
echo "   ngrok will print a public URL like https://xxxx-8080.ngrok-free.app"
echo "   which you can share with students for short-lived demos."
echo "----------------------------------------------------------------"
