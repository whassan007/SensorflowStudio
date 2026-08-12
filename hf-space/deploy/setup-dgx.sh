#!/bin/bash
# Sensorflow DGX Spark Deployment Setup
# Run this on the DGX Spark machine

set -e

APP_BASE="/opt/sensorflow"
STUDIO_DIR="$APP_BASE/studio"
ANALYSIS_DIR="$APP_BASE/analysis"
REPO_URL="${1:-$(cd /tmp && pwd)}"

echo "🚀 Sensorflow Deployment to DGX Spark"
echo "======================================="
echo "Base directory: $APP_BASE"
echo "Repo source: $REPO_URL"

# Create application user if needed
if ! id "sensorflow" &>/dev/null; then
    echo "Creating sensorflow system user..."
    sudo useradd -r -s /bin/bash -d $APP_BASE sensorflow
fi

# Setup directories
echo "Setting up directories..."
sudo mkdir -p $STUDIO_DIR $ANALYSIS_DIR
sudo chown -R sensorflow:sensorflow $APP_BASE

# Copy code
echo "Copying application code..."
sudo cp -r "$REPO_URL"/* $STUDIO_DIR/ 2>/dev/null || true
mkdir -p $ANALYSIS_DIR
sudo cp $STUDIO_DIR/main.py $ANALYSIS_DIR/

# Create Python virtual environments
echo "Creating virtual environments..."
sudo -u sensorflow python3 -m venv $STUDIO_DIR/venv
sudo -u sensorflow python3 -m venv $ANALYSIS_DIR/venv

# Install dependencies
echo "Installing Python dependencies..."
sudo -u sensorflow $STUDIO_DIR/venv/bin/pip install --upgrade pip setuptools wheel
sudo -u sensorflow $STUDIO_DIR/venv/bin/pip install -r $STUDIO_DIR/requirements-prod.txt
sudo -u sensorflow $ANALYSIS_DIR/venv/bin/pip install -r $STUDIO_DIR/requirements-prod.txt

# Download models
echo "Downloading YOLOv8 models..."
sudo -u sensorflow bash $STUDIO_DIR/deploy/download-models.sh $STUDIO_DIR

# Build React frontend
echo "Building React frontend..."
cd $STUDIO_DIR
if [ -f "package.json" ]; then
    npm install
    npm run build
    echo "Frontend built at $STUDIO_DIR/dist"
fi

# Setup systemd services
echo "Installing systemd services..."
sudo cp $STUDIO_DIR/deploy/sensorflow-studio.service /etc/systemd/system/
sudo cp $STUDIO_DIR/deploy/accident-analysis.service /etc/systemd/system/
sudo systemctl daemon-reload

# Setup nginx
echo "Configuring nginx..."
sudo cp $STUDIO_DIR/deploy/nginx.conf /etc/nginx/sites-available/sensorflow.conf
sudo ln -sf /etc/nginx/sites-available/sensorflow.conf /etc/nginx/sites-enabled/ 2>/dev/null || true
sudo nginx -t
sudo systemctl restart nginx

# Start services
echo "Starting services..."
sudo systemctl enable sensorflow-studio accident-analysis
sudo systemctl start sensorflow-studio accident-analysis
sudo systemctl status sensorflow-studio accident-analysis

echo ""
echo "✅ Deployment Complete!"
echo "======================================="
echo "Services:"
echo "  - Sensorflow Studio: http://sensorflow.wael.bot/api/studio"
echo "  - Accident Analysis: http://sensorflow.wael.bot/api/analysis"
echo "  - Frontend: http://sensorflow.wael.bot"
echo ""
echo "Monitor logs with:"
echo "  sudo journalctl -u sensorflow-studio -f"
echo "  sudo journalctl -u accident-analysis -f"
