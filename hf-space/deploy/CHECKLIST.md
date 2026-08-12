# Pre-Deployment & Post-Deployment Checklist

## Pre-Deployment (Local Machine)

- [x] Cleanup complete:
  - [x] Removed test files (test_*.py)
  - [x] Removed examples (example_*.py)
  - [x] Removed virtual environments (.venv/)
  - [x] Removed large model files (*.pt) - will download on DGX
  - [x] Removed redundant analysis scripts
  - [x] Removed documentation clutter

- [x] Files prepared:
  - [x] .gitignore created (proper exclusions)
  - [x] requirements-prod.txt created
  - [x] LICENSE file present
  - [x] Deployment scripts created

- [x] Git status:
  - [x] All changes staged
  - [x] Ready to commit

## Deployment Steps

### Step 1: Transfer Code to DGX
```bash
rsync -avz --exclude='.venv' --exclude='node_modules' \
  /path/to/DrivingRepo/ \
  <dgx-host>:/tmp/sensorflow-deploy/
```

### Step 2: Execute Deployment on DGX
```bash
ssh <dgx-host>
cd /tmp/sensorflow-deploy
sudo bash deploy/setup-dgx.sh /tmp/sensorflow-deploy
```

### Step 3: DNS Configuration (GoDaddy)
- [ ] Add A record: sensorflow.wael.bot → [DGX IP]
- [ ] Set TTL to 3600
- [ ] Wait for DNS propagation (5-10 minutes)

## Post-Deployment (Verify on DGX)

### Services
- [ ] Check sensorflow-studio service: `sudo systemctl status sensorflow-studio`
- [ ] Check accident-analysis service: `sudo systemctl status accident-analysis`
- [ ] Check nginx: `sudo systemctl status nginx`

### API Health Checks
```bash
# From DGX or via Tailscale
curl http://sensorflow.wael.bot/
curl http://sensorflow.wael.bot/api/studio/docs
curl http://sensorflow.wael.bot/api/analysis/docs
curl http://sensorflow.wael.bot/health/studio
curl http://sensorflow.wael.bot/health/analysis
```

### Logs Verification
```bash
sudo journalctl -u sensorflow-studio -n 20
sudo journalctl -u accident-analysis -n 20
sudo tail -20 /var/log/nginx/access.log
```

### Data & Models
- [ ] YOLOv8 models downloaded: `ls -lh /opt/sensorflow/studio/yolov8*.pt`
- [ ] React frontend built: `ls -la /opt/sensorflow/studio/dist/`
- [ ] Data directory present: `ls /opt/sensorflow/studio/data/`

### Performance
```bash
# Check resource usage
nvidia-smi  # GPU load
ps aux | grep sensorflow
top -u sensorflow
```

## Maintenance & Monitoring

### Daily
- [ ] Monitor error logs: `sudo journalctl -u sensorflow-studio --since="1 hour ago"`
- [ ] Check disk usage: `df -h /opt/sensorflow/`
- [ ] Verify API responding: `curl http://sensorflow.wael.bot/health/studio`

### Weekly
- [ ] Review full logs: `sudo journalctl -u sensorflow-studio -p warning`
- [ ] Check model accuracy: Review `/opt/sensorflow/studio/runs/detect/`
- [ ] Backup database (if applicable)

### Monthly
- [ ] Review performance metrics
- [ ] Update dependencies: `pip list --outdated`
- [ ] Test disaster recovery

## Troubleshooting Quick Reference

| Issue | Command |
|-------|---------|
| Service won't start | `sudo journalctl -u sensorflow-studio -n 50` |
| Port conflict | `sudo lsof -i :8001` or `:8002` |
| Nginx error | `sudo nginx -t && sudo systemctl restart nginx` |
| DNS not resolving | `nslookup sensorflow.wael.bot` |
| High memory usage | `ps aux --sort=-%mem \| grep sensorflow` |
| Can't access API | Check firewall: `sudo ufw status` |
| Models not found | `bash /opt/sensorflow/studio/deploy/download-models.sh /opt/sensorflow/studio` |

## Rollback Procedure

If deployment fails:

1. Stop services:
   ```bash
   sudo systemctl stop sensorflow-studio accident-analysis
   ```

2. Remove deployment:
   ```bash
   sudo rm -rf /opt/sensorflow
   ```

3. Re-run deployment with fixed parameters:
   ```bash
   sudo bash /tmp/sensorflow-deploy/deploy/setup-dgx.sh /tmp/sensorflow-deploy
   ```

## Completion Checklist

- [ ] DNS resolves to DGX
- [ ] Frontend accessible at sensorflow.wael.bot
- [ ] Studio API responding at /api/studio
- [ ] Analysis API responding at /api/analysis
- [ ] WebSocket connection works for telemetry
- [ ] Models downloaded and verified
- [ ] Services auto-restart on failure
- [ ] Logs are being captured
- [ ] Performance is acceptable
- [ ] Documentation updated (this file)
