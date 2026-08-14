# Sensorflow Deployment Guide - DGX Spark

## Overview

This deployment sets up two persistent FastAPI services on your DGX Spark machine via Tailscale:
- **Sensorflow Studio**: YOLOv8 training and inference service (port 8001)
- **Accident Analysis Platform**: Data analysis and WebSocket API (port 8002)
- **Frontend**: React dashboard via Vite (served at `/`)

All services are reverse-proxied through nginx at `sensorflow.wael.bot`

## Prerequisites

- DGX Spark machine accessible via Tailscale
- SSH access configured (credentials in local SSH config)
- Ubuntu/Debian-based Linux on DGX
- sudo privileges on DGX
- Node.js 18+ (for frontend build)
- Python 3.9+

## Quick Start

### 1. Prepare Code on Local Machine

```bash
cd /path/to/DrivingRepo
# Cleanup already done, verify:
ls -la | grep -v "test_\|example_\|\.venv\|yolov8.*\.pt"
```

### 2. Transfer Code to DGX via Tailscale SSH

```bash
# Using your SSH config
rsync -avz --exclude='.venv' --exclude='node_modules' \
  /path/to/DrivingRepo/ \
  <your-dgx-host>:/tmp/sensorflow-deploy/
```

Replace `<your-dgx-host>` with your DGX Tailscale hostname or IP.

### 3. Run Deployment on DGX

```bash
# SSH into DGX
ssh <your-dgx-host>

# Run deployment script
cd /tmp/sensorflow-deploy
chmod +x deploy/setup-dgx.sh
sudo bash deploy/setup-dgx.sh /tmp/sensorflow-deploy
```

### 4. Configure DNS (GoDaddy)

Add an A record pointing to your DGX Spark's external IP or Tailscale IP:
- **Domain**: sensorflow.wael.bot
- **Type**: A
- **Value**: Your DGX Spark's IP
- **TTL**: 3600

If using Tailscale:
- Get your Tailscale IP: `tailscale ip -4`
- Use this IP for the A record

## Service Management

Check service status:
```bash
sudo systemctl status sensorflow-studio accident-analysis
```

View logs:
```bash
# Studio backend
sudo journalctl -u sensorflow-studio -f

# Analysis backend
sudo journalctl -u accident-analysis -f

# Nginx
sudo tail -f /var/log/nginx/error.log
```

Restart services:
```bash
sudo systemctl restart sensorflow-studio accident-analysis
```

Stop services:
```bash
sudo systemctl stop sensorflow-studio accident-analysis
```

## API Endpoints

### Studio (YOLOv8)
- `POST /api/studio/train` - Start training
- `GET /api/studio/train/logs` - Stream training logs
- `POST /api/studio/infer` - Run inference
- `POST /api/studio/export` - Export model
- Interactive docs: `/api/studio/docs`

### Analysis (Accident Data)
- `POST /api/analysis/upload` - Upload accident data
- `GET /api/analysis/analyze` - Analyze data
- `WS /ws/telemetry` - Real-time telemetry WebSocket
- Interactive docs: `/api/analysis/docs`

### Frontend
- `GET /` - React dashboard

## Directory Structure on DGX

```
/opt/sensorflow/
├── studio/
│   ├── app_backend.py
│   ├── main.py
│   ├── venv/              # Python venv
│   ├── dist/              # Built React app
│   ├── src/               # React source
│   ├── requirements-prod.txt
│   └── deploy/
├── analysis/
│   ├── main.py
│   ├── venv/
│   └── accident_*.py
└── models/
    ├── yolov8n.pt
    └── yolov8m.pt
```

## Monitoring

Check if services are responding:
```bash
curl http://sensorflow.wael.bot/health/studio
curl http://sensorflow.wael.bot/health/analysis
curl http://sensorflow.wael.bot/
```

Monitor system resources:
```bash
# On DGX
nvidia-smi  # GPU usage
top -u sensorflow  # CPU/Memory
```

## Troubleshooting

### Services fail to start
```bash
sudo journalctl -u sensorflow-studio -n 50
sudo systemctl status sensorflow-studio
```

### Port conflicts
```bash
sudo lsof -i :8001  # Check port 8001
sudo lsof -i :8002  # Check port 8002
```

### Nginx issues
```bash
sudo nginx -t  # Test config
sudo systemctl restart nginx
tail -f /var/log/nginx/error.log
```

### Model download failed
```bash
# Manual download
cd /opt/sensorflow/studio
bash deploy/download-models.sh .
```

## Updates & Maintenance

To update code:
```bash
# On DGX
cd /opt/sensorflow/studio
git pull  # if using git
# or rsync again from local machine

# Restart services
sudo systemctl restart sensorflow-studio accident-analysis
```

## License

This project is released under the **PolyForm Noncommercial License 1.0.0**
(source-available / non-commercial; not OSI Open Source).
See `LICENSE` and `NOTICE` for details. Commercial use requires a separate license.

## Support

For issues or questions, check:
1. Service logs: `sudo journalctl -u sensorflow-studio`
2. Nginx config: `/etc/nginx/sites-available/sensorflow.conf`
3. Python venv activation: `/opt/sensorflow/studio/venv/bin/python`
