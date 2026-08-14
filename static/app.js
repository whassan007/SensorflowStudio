document.addEventListener('DOMContentLoaded', () => {
    // API base URL
    const API_BASE = '';

    // Non-blocking status signaling (replaces window.alert for routine feedback)
    const toastStack = document.getElementById('toast-stack');
    const TOAST_ICONS = { success: '✓', error: '✕', warning: '!', info: 'ℹ' };
    const TOAST_DEFAULT_MS = { success: 3800, info: 4200, warning: 5200, error: 6500 };

    function notify(message, type = 'info', options = {}) {
        if (!toastStack || message == null || message === '') return;
        const kind = ['success', 'error', 'warning', 'info'].includes(type) ? type : 'info';
        const duration = options.duration ?? TOAST_DEFAULT_MS[kind];

        const el = document.createElement('div');
        el.className = `toast toast-${kind}`;
        el.setAttribute('role', kind === 'error' ? 'alert' : 'status');

        const icon = document.createElement('span');
        icon.className = 'toast-icon';
        icon.textContent = TOAST_ICONS[kind];
        icon.setAttribute('aria-hidden', 'true');

        const body = document.createElement('div');
        body.className = 'toast-body';
        body.textContent = String(message);

        const close = document.createElement('button');
        close.type = 'button';
        close.className = 'toast-close';
        close.setAttribute('aria-label', 'Dismiss');
        close.textContent = '×';

        let hideTimer = null;
        const dismiss = () => {
            if (hideTimer) clearTimeout(hideTimer);
            el.classList.add('toast-out');
            setTimeout(() => el.remove(), 180);
        };
        close.addEventListener('click', dismiss);

        el.appendChild(icon);
        el.appendChild(body);
        el.appendChild(close);
        toastStack.appendChild(el);

        if (duration > 0) {
            hideTimer = setTimeout(dismiss, duration);
        }
        return dismiss;
    }

    /**
     * Continuous UI metrics: integers stay whole; non-integers use up to 2 decimals.
     */
    function formatNumber(n) {
        if (n === null || n === undefined || n === '') return '—';
        const v = typeof n === 'number' ? n : Number(n);
        if (!Number.isFinite(v)) return String(n);
        if (Number.isInteger(v)) return String(v);
        const rounded = Math.round(v * 100) / 100;
        if (Number.isInteger(rounded)) return String(rounded);
        return parseFloat(rounded.toFixed(2)).toString();
    }

    /**
     * Percentages: prefer one decimal like 10.0%; never more than 2.
     * Fractions in [-1, 1] are scaled ×100 unless alreadyPercent is set.
     */
    function formatPercent(n, opts = {}) {
        if (n === null || n === undefined || n === '') return '—';
        let pct = typeof n === 'number' ? n : Number(n);
        if (!Number.isFinite(pct)) return String(n);
        const already = opts.alreadyPercent === true || Math.abs(pct) > 1;
        if (!already) pct *= 100;
        const digits = opts.digits === 2 ? 2 : 1;
        const factor = 10 ** digits;
        return `${(Math.round(pct * factor) / factor).toFixed(digits)}%`;
    }

    /** Format evidence/console metric values; percent-like keys get formatPercent. */
    function formatMetricValue(key, value) {
        if (value === null || value === undefined) return '—';
        if (typeof value === 'boolean') return value ? 'true' : 'false';
        if (typeof value === 'object') return JSON.stringify(value);
        if (typeof value === 'string') {
            const m = value.trim().match(/^(-?\d+(?:\.\d+)?)\s*%$/);
            if (m) return formatPercent(Number(m[1]), { alreadyPercent: true });
            return value;
        }
        if (typeof value !== 'number' || !Number.isFinite(value)) return String(value);
        const k = String(key || '').toLowerCase();
        if (/(?:^|_)pct(?:_|$)|percent|percentage|_rate$|loaded_pct/.test(k)) {
            return formatPercent(value, { alreadyPercent: Math.abs(value) > 1 || /_pct|percent/.test(k) });
        }
        return formatNumber(value);
    }

    // Stage Navigation
    const navButtons = document.querySelectorAll('.nav-btn');
    const panels = document.querySelectorAll('.stage-panel');
    const stageTitle = document.getElementById('stage-title');
    const stageDesc = document.getElementById('stage-desc');
    const nextBtn = document.getElementById('btn-global-action');

    const STAGE_METADATA = {
        dataset: { title: 'Dataset Configuration', desc: 'Prepare and validate your image source path and annotation files.', next: 'ingest', nextLabel: 'Proceed to 3D Ingest' },
        ingest: { title: '3D Ingest & Fusion', desc: 'Fuse Alpamayo/Waymo/A2D2 stubs and/or local image/video sequences with six-axis taxonomy stratification.', next: 'perception', nextLabel: 'Proceed to 3D Perception', gateKey: 'ingest_complete', completionOnly: true },
        perception: { title: '3D Perception', desc: 'SAM-based 2D masks lifted to 3D bounding box proposals.', next: 'tracking', nextLabel: 'Proceed to Tracking', gateKey: 'perception_complete', completionOnly: true },
        tracking: { title: 'Temporal Tracking', desc: 'Kalman + Hungarian association for ID-smooth multi-frame tracks.', next: 'quality-gate', nextLabel: 'Proceed to Quality Gate', gateKey: 'tracking_complete', completionOnly: true },
        'quality-gate': { title: 'Quality Gate', desc: 'Benchmark automated tracks against vendor GT with 3D and temporal metrics.', next: 'launch-gate', nextLabel: 'Proceed to Launch Gate', gateKey: 'quality_gate_passed', completionKey: 'benchmark_complete' },
        'launch-gate': { title: 'Launch Gate', desc: 'Validate safety thresholds before export is allowed.', next: 'model', nextLabel: 'Proceed to Model Setup', gateKey: 'launch_gate_passed' },
        model: { title: 'Model Setup', desc: 'Choose a YOLOv8 base model and compute device architecture.', next: 'training', nextLabel: 'Proceed to Training' },
        training: { title: 'Training Execution', desc: 'Fine-tune the selected YOLO model on your configured dataset.', next: 'inference', nextLabel: 'Proceed to Inference' },
        inference: { title: 'Auto-Labeler Inference', desc: 'Run predictions on raw images using your fine-tuned weights.', next: 'grader', nextLabel: 'Proceed to Auto-Grader' },
        grader: { title: 'Auto-Grader Quality', desc: 'Scan predictions for anomalies, confidence scores, and label quality.', next: 'mitl', nextLabel: 'Proceed to MITL Review' },
        mitl: { title: 'Man-in-the-Loop Review', desc: 'Curate predictions, telemetry, and Chain-of-Causation reasoning traces.', next: 'benchmark', nextLabel: 'Proceed to Benchmarking' },
        benchmark: { title: 'Model Benchmarking', desc: 'Compare standard vs. VLM autonomous driving models on safety-critical metrics.', next: 'export', nextLabel: 'Proceed to Export' },
        export: { title: 'Export & Deploy', desc: 'Serialize model weights into production formats like ONNX.', next: 'mcp', nextLabel: 'Proceed to MCP Settings' },
        mcp: { title: 'MCP Settings', desc: 'View, edit, and toggle Model Context Protocol servers in mcp_config.json.', next: 'ssam', nextLabel: 'Proceed to SSAM Safety' },
        ssam: { title: 'SSAM Safety', desc: 'View street conflict metrics and add automated or manual severity annotations.', next: 'dataset', nextLabel: 'Back to Start' }
    };

    let pipelineState = {};
    let currentSequenceId = 'seq_001';

    async function refreshPipelineStatus() {
        try {
            const res = await fetch(`${API_BASE}/api/pipeline/status?sequence_id=${currentSequenceId}`);
            const data = await res.json();
            pipelineState = data;
            updateGateBadges();
        } catch (e) { /* backend offline */ }
    }

    function updateGateBadges() {
        navButtons.forEach(btn => {
            const stage = btn.dataset.stage;
            const meta = STAGE_METADATA[stage];
            if (!meta || !meta.gateKey) return;
            let badge = btn.querySelector('.gate-badge');
            if (!badge) {
                badge = document.createElement('span');
                badge.className = 'gate-badge';
                badge.style.cssText = 'margin-left:6px;font-size:10px;padding:2px 6px;border-radius:4px;';
                btn.appendChild(badge);
            }

            // Completion-only stages: PASS when done, blank when not run (never FAIL).
            // Pass/fail gates (quality/launch): PASS/FAIL only after evaluation; blank beforehand.
            const verdict = pipelineState[meta.gateKey];
            const completed = meta.completionKey ? pipelineState[meta.completionKey] : null;

            if (meta.completionOnly) {
                if (verdict === true) {
                    badge.textContent = pipelineState.demo_stub ? 'STUB' : 'DONE';
                    badge.style.background = pipelineState.demo_stub ? '#fbbf2433' : '#00ffaa33';
                    badge.style.color = pipelineState.demo_stub ? '#fbbf24' : '#00ffaa';
                } else {
                    badge.textContent = '';
                }
                return;
            }

            if (verdict === true) {
                badge.textContent = 'PASS';
                badge.style.background = '#00ffaa33';
                badge.style.color = '#00ffaa';
            } else if (verdict === false) {
                badge.textContent = 'FAIL';
                badge.style.background = '#ff444433';
                badge.style.color = '#ff4444';
            } else if (completed === true) {
                badge.textContent = 'RAN';
                badge.style.background = '#3b82f633';
                badge.style.color = '#93c5fd';
            } else {
                badge.textContent = '';
            }
        });
    }

    refreshPipelineStatus();

    function switchStage(stageId) {
        // Update nav buttons
        navButtons.forEach(btn => {
            if (btn.dataset.stage === stageId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update panels
        panels.forEach(panel => {
            if (panel.id === `panel-${stageId}`) {
                panel.classList.add('active');
            } else {
                panel.classList.remove('active');
            }
        });

        // Update header metadata
        const meta = STAGE_METADATA[stageId];
        stageTitle.textContent = meta.title;
        stageDesc.textContent = meta.desc;
        nextBtn.textContent = meta.nextLabel;
        nextBtn.dataset.next = meta.next;
    }

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            switchStage(btn.dataset.stage);
        });
    });

    nextBtn.addEventListener('click', () => {
        const currentStage = document.querySelector('.nav-btn.active')?.dataset.stage;
        const meta = STAGE_METADATA[currentStage];
        if (meta && meta.gateKey && pipelineState[meta.gateKey] === false) {
            notify('Launch gate not passed. Complete the quality gate first.', 'warning');
            return;
        }
        switchStage(nextBtn.dataset.next);
    });

    // Model selection
    const modelCards = document.querySelectorAll('.model-card');
    const modelDisplay = document.getElementById('selected-weights-display');
    let selectedModel = 'yolov8n.pt';

    modelCards.forEach(card => {
        card.addEventListener('click', () => {
            modelCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');
            selectedModel = card.dataset.model;
            modelDisplay.textContent = selectedModel;
        });
    });

    // Subtabs within Training panel
    const subtabButtons = document.querySelectorAll('.tab-sub-btn');
    const subtabContents = document.querySelectorAll('.tab-sub-content');

    subtabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            subtabButtons.forEach(b => b.classList.remove('active'));
            subtabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`subtab-${btn.dataset.tab}`).classList.add('active');
        });
    });

    // Save Dataset Configuration
    const btnSaveDataset = document.getElementById('btn-save-dataset');
    const dataYamlPath = document.getElementById('data-yaml-path');
    const datasetSourcePath = document.getElementById('dataset-source-path');

    btnSaveDataset.addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE}/api/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    yaml_path: dataYamlPath.value,
                    source_path: datasetSourcePath.value
                })
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            notify('Configuration saved (YAML/paths not yet executed).', 'success');
            const cfgStatus = document.getElementById('config-exec-status');
            if (cfgStatus) {
                cfgStatus.textContent = 'CONFIGURATION SAVED / not yet used by an execution — run Load & Preprocess.';
                cfgStatus.dataset.modified = '1';
            }
        } catch (e) {
            console.error(e);
            notify('Failed to save dataset configuration.', 'error');
        }
    });

    // Run Pre-Check
    document.getElementById('btn-validate').addEventListener('click', async () => {
        try {
            const res = await fetch(`${API_BASE}/api/precheck`);
            const data = await res.json();
            const precheckType = data.verified ? 'success' : 'warning';
            notify(`Pre-check: ${data.status}\n${data.message || ''}`, precheckType);
            refreshScriptStatus(data.scripts);
        } catch (e) {
            notify('Failed to complete pre-check.', 'error');
        }
    });

    // Device Selection Change Listener
    const selectDevice = document.getElementById('select-device');
    const dgxSetupInstructions = document.getElementById('dgx-setup-instructions');

    selectDevice.addEventListener('change', () => {
        if (selectDevice.value === 'dgx-spark') {
            dgxSetupInstructions.classList.remove('hidden');
        } else {
            dgxSetupInstructions.classList.add('hidden');
        }
    });

    // Training state and polling
    let trainInterval = null;
    const btnStartTrain = document.getElementById('btn-start-train');
    const btnStopTrain = document.getElementById('btn-stop-train');
    const trainProgressSection = document.getElementById('train-progress-section');
    const trainProgressBar = document.getElementById('train-progress-bar');
    const progressPercent = document.getElementById('progress-percent');
    const terminalLogs = document.getElementById('terminal-logs');

    btnStartTrain.addEventListener('click', async () => {
        const epochs = document.getElementById('train-epochs').value;
        const batch = document.getElementById('train-batch').value;
        const device = document.getElementById('select-device').value;

        terminalLogs.textContent = 'Launching training process...\n';
        btnStartTrain.classList.add('hidden');
        btnStopTrain.classList.remove('hidden');
        trainProgressSection.classList.remove('hidden');

        try {
            const res = await fetch(`${API_BASE}/api/train/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel,
                    epochs: parseInt(epochs),
                    batch: parseInt(batch),
                    device: device,
                    data: dataYamlPath.value
                })
            });
            const data = await res.json();
            if (!res.ok) {
                terminalLogs.textContent += `Error: ${JSON.stringify(data)}\n`;
                resetTrainingUI();
                notify(data.detail?.message || data.detail || 'Failed to start training', 'error');
                return;
            }
            renderEvidenceCard('evidence-training', {
                status: 'RUNNING',
                execution_id: data.execution_id,
                process_id: data.process_id,
                command: data.command,
                message: 'Training process spawned',
            });
            pollTraining();
            trainInterval = setInterval(pollTraining, 1500);
        } catch (e) {
            terminalLogs.textContent += `Error starting training: ${e.message}\n`;
            resetTrainingUI();
        }
    });

    btnStopTrain.addEventListener('click', async () => {
        try {
            await fetch(`${API_BASE}/api/train/stop`, { method: 'POST' });
            terminalLogs.textContent += '\n--- Training terminated by user ---\n';
        } catch (e) {
            console.error(e);
        }
        resetTrainingUI();
    });

    function resetTrainingUI() {
        if (trainInterval) {
            clearInterval(trainInterval);
            trainInterval = null;
        }
        btnStartTrain.classList.remove('hidden');
        btnStopTrain.classList.add('hidden');
    }

    async function pollTraining() {
        try {
            const res = await fetch(`${API_BASE}/api/train/status`);
            const status = await res.json();

            // Append logs
            if (status.logs) {
                terminalLogs.textContent = status.logs;
                terminalLogs.scrollTop = terminalLogs.scrollHeight;
            }

            // Update progress
            const pct = Math.min(100, Math.max(0, (status.progress <= 1 ? status.progress * 100 : status.progress)));
            trainProgressBar.style.width = `${pct}%`;
            progressPercent.textContent = formatPercent(pct, { alreadyPercent: true });

            // Draw loss curves if we have values
            if (status.losses && status.losses.length > 0) {
                drawLossChart(status.losses);
            }

            if (!status.running) {
                const verdict = status.status || (status.exit_code === 0 ? 'SUCCEEDED' : 'FAILED');
                terminalLogs.textContent += `\n--- Process finished: ${verdict} exit_code=${status.exit_code} ---\n`;
                renderEvidenceCard('evidence-training', {
                    status: verdict,
                    execution_id: status.execution_id,
                    exit_code: status.exit_code,
                    process_id: status.process_id,
                    command: status.command,
                    metrics: {
                        epochs_observed: status.epochs_observed,
                        epochs_requested: status.epochs_requested,
                        checkpoint: status.checkpoint,
                        losses_parsed: (status.losses || []).length,
                    },
                    message: `Training ${verdict}`,
                });
                refreshExecutionConsole();
                resetTrainingUI();
            }
        } catch (e) {
            console.error(e);
        }
    }

    // Canvas Loss Curve drawer (Zero external dependencies)
    function drawLossChart(losses) {
        const canvas = document.getElementById('loss-chart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background
        ctx.fillStyle = '#04060b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Margins
        const margin = 40;
        const w = canvas.width - margin * 2;
        const h = canvas.height - margin * 2;

        const maxLoss = Math.max(...losses, 1.0);
        const minLoss = 0;

        // Draw grid lines
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = margin + (h / 4) * i;
            ctx.beginPath();
            ctx.moveTo(margin, y);
            ctx.lineTo(canvas.width - margin, y);
            ctx.stroke();

            // Y label
            ctx.fillStyle = '#6b7280';
            ctx.font = '10px sans-serif';
            ctx.fillText(formatNumber(maxLoss - (maxLoss / 4) * i), 10, y + 4);
        }

        // Draw line
        if (losses.length < 2) return;

        ctx.strokeStyle = '#00c896';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(0, 200, 150, 0.4)';
        ctx.shadowBlur = 8;
        ctx.beginPath();

        losses.forEach((loss, i) => {
            const x = margin + (w / (losses.length - 1)) * i;
            const y = margin + h - (loss / maxLoss) * h;
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset shadow

        // Draw points
        ctx.fillStyle = '#3b82f6';
        losses.forEach((loss, i) => {
            const x = margin + (w / (losses.length - 1)) * i;
            const y = margin + h - (loss / maxLoss) * h;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });

        // X labels
        ctx.fillStyle = '#6b7280';
        ctx.fillText('Epochs progress', canvas.width / 2 - 30, canvas.height - 10);
    }

    // Auto-Labeler Inference
    const btnRunInfer = document.getElementById('btn-run-infer');
    const selectInferImage = document.getElementById('select-infer-image');
    const predictionImg = document.getElementById('prediction-img');
    const valInferConf = document.getElementById('val-infer-conf');
    const valInferIou = document.getElementById('val-infer-iou');

    document.getElementById('infer-conf').addEventListener('input', (e) => {
        valInferConf.textContent = formatNumber(parseFloat(e.target.value));
    });

    document.getElementById('infer-iou').addEventListener('input', (e) => {
        valInferIou.textContent = formatNumber(parseFloat(e.target.value));
    });

    btnRunInfer.addEventListener('click', async () => {
        btnRunInfer.textContent = 'Predicting...';
        btnRunInfer.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/infer/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    weights: document.getElementById('infer-weights').value,
                    conf: parseFloat(document.getElementById('infer-conf').value),
                    iou: parseFloat(document.getElementById('infer-iou').value),
                    source: datasetSourcePath.value
                })
            });
            const data = await res.json();
            if (!res.ok) {
                const detail = data.detail?.message || data.detail || JSON.stringify(data);
                notify(`Inference failed: ${detail}`, 'error');
                renderEvidenceCard('evidence-inference', {
                    status: 'FAILED',
                    execution_id: data.detail?.execution_id || data.execution_id,
                    message: detail,
                });
                return;
            }

            selectInferImage.innerHTML = '';
            if (data.images && data.images.length > 0) {
                data.images.forEach(img => {
                    const opt = document.createElement('option');
                    opt.value = img;
                    opt.textContent = img;
                    selectInferImage.appendChild(opt);
                });
                loadImage(data.images[0]);
            } else {
                selectInferImage.innerHTML = '<option value="">No images found</option>';
            }
            renderEvidenceCard('evidence-inference', {
                status: data.status,
                execution_id: data.execution_id,
                duration_ms: data.duration_ms,
                model: data.model,
                checkpoint: data.checkpoint,
                metrics: {
                    discovered: data.records_discovered,
                    processed: data.records_processed,
                    succeeded: data.records_succeeded,
                    failed: data.records_failed,
                    inference_calls: data.inference_calls,
                    predictions: data.predictions_generated,
                    output_dir: data.output_dir,
                    exit_code: data.exit_code,
                },
                message: data.message,
            });
            refreshExecutionConsole();
            if (data.status === 'SUCCEEDED' || data.status === 'PARTIAL_SUCCESS') {
                notify(`Inference ${data.status}: ${data.records_succeeded}/${data.records_discovered} images`, 'success');
            } else {
                notify(`Inference ${data.status}: ${data.message || 'see evidence'}`, 'warning');
            }
        } catch (e) {
            console.error(e);
            notify('Inference execution failed.', 'error');
        } finally {
            btnRunInfer.textContent = '⚡ Run Auto-Labeler';
            btnRunInfer.disabled = false;
        }
    });

    selectInferImage.addEventListener('change', (e) => {
        if (e.target.value) {
            loadImage(e.target.value);
        }
    });

    function loadImage(filename) {
        predictionImg.src = `${API_BASE}/api/images/${filename}?t=${Date.now()}`;
    }

    // Auto-Grader Quality Diagnostics
    const btnRunGrader = document.getElementById('btn-run-grader');
    const graderScore = document.getElementById('grader-score');
    const statTotalImgs = document.getElementById('stat-total-imgs');
    const statTotalPreds = document.getElementById('stat-total-preds');
    const graderIssuesList = document.getElementById('grader-issues-list');

    btnRunGrader.addEventListener('click', async () => {
        btnRunGrader.textContent = 'Scanning...';
        btnRunGrader.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/grade`);
            const data = await res.json();
            if (!res.ok) {
                notify(data.detail?.message || data.detail || 'Grader failed', 'error');
                renderEvidenceCard('evidence-grader', {
                    status: 'FAILED',
                    execution_id: data.detail?.execution_id || data.execution_id,
                    message: data.detail?.message || data.detail,
                });
                return;
            }

            graderScore.textContent = data.quality_score != null ? formatNumber(Number(data.quality_score)) : '—';
            statTotalImgs.textContent = data.total_images ?? '—';
            statTotalPreds.textContent = data.total_predictions ?? '—';

            graderIssuesList.innerHTML = '';
            if (data.status === 'NOT_EXECUTED') {
                graderIssuesList.innerHTML = '<div class="empty-state">NOT_EXECUTED — no predictions artifact. This is not a perfect score.</div>';
            } else if (data.issues && data.issues.length > 0) {
                data.issues.forEach(issue => {
                    const el = document.createElement('div');
                    el.className = `issue-card ${issue.severity.toLowerCase()}`;
                    el.innerHTML = `
                        <h5>${issue.type} (${issue.severity})</h5>
                        <p>${issue.description}</p>
                        <div class="rec">💡 Recommendation: ${issue.recommendation}</div>
                    `;
                    graderIssuesList.appendChild(el);
                });
            } else {
                graderIssuesList.innerHTML = '<div class="empty-state">No quality anomalies reported by grader output (not a claim of ground-truth perfection).</div>';
            }
            renderEvidenceCard('evidence-grader', {
                status: data.status,
                execution_id: data.execution_id,
                duration_ms: data.duration_ms,
                metrics: data.metrics || {
                    total_images: data.total_images,
                    total_predictions: data.total_predictions,
                    quality_score: data.quality_score,
                },
                message: data.message,
            });
            refreshExecutionConsole();
        } catch (e) {
            console.error(e);
            notify('Failed to run quality diagnostics.', 'error');
        } finally {
            btnRunGrader.textContent = 'Run Quality Diagnostics';
            btnRunGrader.disabled = false;
        }
    });

    // ONNX Weight Export
    const btnExportWeights = document.getElementById('btn-export-weights');
    const exportStatusBox = document.getElementById('export-status-box');

    btnExportWeights.addEventListener('click', async () => {
        const format = document.querySelector('input[name="export-format"]:checked').value;
        exportStatusBox.classList.remove('hidden');
        exportStatusBox.textContent = `Compiling YOLO model weights to ${format.toUpperCase()}...`;
        btnExportWeights.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    weights: document.getElementById('infer-weights').value,
                    format: format
                })
            });
            const data = await res.json();
            exportStatusBox.textContent = `✓ Export complete! File generated: ${data.exported_file}`;
        } catch (e) {
            console.error(e);
            exportStatusBox.textContent = `Error during compilation: ${e.message}`;
        } finally {
            btnExportWeights.disabled = false;
        }
    });

    // AI Copilot Triage Assistant
    const btnCopilotAsk = document.getElementById('btn-copilot-ask');
    const copilotContainer = document.getElementById('copilot-response-container');
    const copilotCard = document.querySelector('.copilot-card');

    function parseMarkdown(md) {
        if (!md) return '';
        let html = md
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Fenced code blocks
        html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
            return `<pre class="copilot-code"><code>${code.trim()}</code></pre>`;
        });

        // Headings
        html = html.replace(/^### (.*?)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.*?)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.*?)$/gm, '<h2>$1</h2>');

        // Bullet lists
        html = html.replace(/^\s*[-*+]\s+(.*?)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
        html = html.replace(/<\/ul>\s*<ul>/g, '');

        // Bold text
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Paras
        html = html.split('\n').map(line => {
            const trimmed = line.trim();
            if (!trimmed) return '';
            if (trimmed.startsWith('<h') || trimmed.startsWith('<l') || trimmed.startsWith('<u') || trimmed.startsWith('<p') || trimmed.startsWith('<pre')) {
                return line;
            }
            return `<p>${trimmed}</p>`;
        }).join('\n');

        return html;
    }

    btnCopilotAsk.addEventListener('click', async () => {
        btnCopilotAsk.textContent = 'Analyzing Pipeline (Gemma 4)...';
        btnCopilotAsk.disabled = true;
        copilotCard.classList.add('copilot-loading');
        copilotContainer.innerHTML = '<div class="copilot-spinner-wrapper"><span class="copilot-spinner"></span><span style="margin-left: 10px;">Synthesizing training metrics and triage report...</span></div>';

        try {
            const res = await fetch(`${API_BASE}/api/copilot/explain`, { method: 'POST' });
            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Failed to connect to Ollama');
            }
            const data = await res.json();
            copilotContainer.innerHTML = `
                <div class="copilot-meta">
                    <span class="provider">Active Provider: <code>${data.provider}</code></span>
                </div>
                <div class="copilot-markdown">${parseMarkdown(data.analysis)}</div>
            `;
        } catch (e) {
            console.error(e);
            copilotContainer.innerHTML = `
                <div class="copilot-error">
                    <h5>⚠️ Copilot Analysis Failed</h5>
                    <p>${e.message}</p>
                    <p style="font-size: 11px; margin-top: 5px;">Ensure Ollama is running and has gemma4 loaded, or check your local network connection.</p>
                </div>
            `;
        } finally {
            btnCopilotAsk.textContent = 'Ask AI Triage Copilot';
            btnCopilotAsk.disabled = false;
            copilotCard.classList.remove('copilot-loading');
        }
    });

    // --- MAN-IN-THE-LOOP TRIAGE CONTROLLERS ---
    const selectDatasetType = document.getElementById('select-dataset-type');
    const mitlCameraImg = document.getElementById('mitl-camera-img');
    const mitlCameraLabel = document.getElementById('mitl-camera-label');
    const mitlGps = document.getElementById('mitl-telemetry-gps');
    const mitlSpeed = document.getElementById('mitl-telemetry-speed');
    const mitlAccel = document.getElementById('mitl-telemetry-accel');
    const mitlImu = document.getElementById('mitl-telemetry-imu');
    const mitlBoxRows = document.getElementById('mitl-box-rows');
    const mitlCocTrace = document.getElementById('mitl-coc-trace');
    
    const btnMitlSave = document.getElementById('btn-mitl-save');
    const btnMitlAudit = document.getElementById('btn-mitl-audit');
    const btnMitlAddBox = document.getElementById('btn-mitl-add-box');
    const mitlAuditResults = document.getElementById('mitl-audit-results');
    const mitlAuditText = document.getElementById('mitl-audit-text');

    let currentMitlData = null;
    let originalCocText = "";

    // Load NVIDIA Dataset from HF (Simulated)
    async function loadNvidiaDataset(datasetName) {
        try {
            const res = await fetch(`${API_BASE}/api/nvidia/load`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: datasetName })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                currentMitlData = data.data;
                originalCocText = currentMitlData.coc_trace;
                renderMitlData();
                notify(`Dataset loaded: ${currentMitlData.dataset}`, 'success');
            }
        } catch (e) {
            console.error(e);
            notify('Failed to load NVIDIA dataset.', 'error');
        }
    }

    const datasetLinksContainer = document.getElementById('dataset-source-links');
    const datasetLinksContent = document.getElementById('dataset-links-content');
    const btnPreprocessDataset = document.getElementById('btn-preprocess-dataset');
    const pipelineWizardModal = document.getElementById('pipeline-wizard-modal');
    const btnWizardSubmit = document.getElementById('btn-wizard-submit');

    const DATASET_LINKS = {
        waymo: [
            { text: "Official Portal: waymo.com/open", url: "https://waymo.com/open/" },
            { text: "GitHub Repository: github.com/waymo-research/waymo-open-dataset", url: "https://github.com/waymo-research/waymo-open-dataset" }
        ],
        alpamayo: [
            { text: "Official Platform: nvidia.com/solutions/autonomous-vehicles/alpamayo", url: "https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/" },
            { text: "Dataset Access / GitHub: github.com/NVlabs/physical_ai_av", url: "https://github.com/NVlabs/physical_ai_av" }
        ],
        a2d2: [
            { text: "Official Download Site: a2d2.audi/a2d2/en/download.html", url: "https://www.a2d2.audi/a2d2/en/download.html" },
            { text: "AWS Cloud Hosting: registry.opendata.aws/aev-a2d2", url: "https://registry.opendata.aws/aev-a2d2/" }
        ]
    };

    function renderSourceLinks(datasetType) {
        if (datasetType === 'local') {
            datasetLinksContainer.classList.add('hidden');
            return;
        }
        datasetLinksContainer.classList.remove('hidden');
        datasetLinksContent.innerHTML = '';
        
        const links = DATASET_LINKS[datasetType] || [];
        links.forEach(lnk => {
            const a = document.createElement('a');
            a.href = lnk.url;
            a.target = '_blank';
            a.className = 'dataset-link-item';
            a.innerHTML = `🔗 <strong>${lnk.text}</strong>`;
            datasetLinksContent.appendChild(a);
        });
    }

    selectDatasetType.addEventListener('change', (e) => {
        const val = e.target.value;
        renderSourceLinks(val);
        loadCatalogMetadata(val);
        if (val !== 'local') {
            const backendDatasetMap = {
                waymo: 'physical_ai',
                alpamayo: 'physical_ai',
                a2d2: 'nurec'
            };
            loadNvidiaDataset(backendDatasetMap[val] || 'physical_ai');
        }
    });

    function applyCatalogMetadata(meta, opts = {}) {
        const badge = document.getElementById('ds-meta-badge');
        const desc = document.getElementById('ds-meta-desc');
        const pills = document.getElementById('ds-class-pills');
        if (!meta) return;
        const setText = (id, val) => {
            const el = document.getElementById(id);
            if (el) el.textContent = val ?? '—';
        };
        setText('ds-kpi-total-rows', meta.total_rows_source);
        setText('ds-kpi-loaded-rows', meta.loaded_rows);
        setText('ds-kpi-ingest-pct', meta.ingestion_pct != null ? formatMetricValue('ingestion_pct', meta.ingestion_pct) : '—');
        setText('ds-kpi-total-annotations', meta.total_annotations);
        setText('ds-meta-sensors', meta.sensor_modality);
        setText('ds-meta-geo', meta.geographic_coverage);
        setText('ds-meta-weather', meta.weather_conditions);
        setText('ds-meta-annotator', meta.annotation_tool);
        setText('ds-meta-footprint', meta.storage_footprint);
        setText('ds-meta-format', meta.format);
        setText('ds-meta-license', meta.licensing);
        setText('ds-meta-type-id', meta.dataset_type);
        if (pills) {
            pills.innerHTML = '';
            const counts = meta.class_counts || {};
            Object.entries(counts).forEach(([cls, count]) => {
                const span = document.createElement('span');
                span.className = 'badge';
                span.style.cssText = 'background:rgba(56,189,248,0.12);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);padding:4px 8px;font-size:11px;';
                span.textContent = `${cls}: ${count}`;
                pills.appendChild(span);
            });
        }
        if (desc && meta.browse_hint) {
            desc.textContent = meta.browse_hint;
        }
        if (badge) {
            if (opts.browsable) {
                badge.textContent = opts.badgeText || 'Browsable on disk';
                badge.style.background = 'rgba(0, 255, 170, 0.15)';
                badge.style.color = '#00ffaa';
                badge.style.border = '1px solid rgba(0, 255, 170, 0.3)';
            } else {
                badge.textContent = opts.badgeText || `${formatMetricValue('ingestion_pct', meta.ingestion_pct || '—')} catalog — not browsable`;
                badge.style.background = 'rgba(251, 191, 36, 0.15)';
                badge.style.color = '#fbbf24';
                badge.style.border = '1px solid rgba(251, 191, 36, 0.35)';
            }
        }
    }

    async function loadCatalogMetadata(datasetType) {
        try {
            const res = await fetch(`${API_BASE}/api/dataset/details?type=${encodeURIComponent(datasetType || 'local')}`);
            const data = await res.json();
            if (data.status === 'ok') applyCatalogMetadata(data.metadata, { browsable: false });
        } catch (e) { /* backend offline */ }
    }

    btnPreprocessDataset.addEventListener('click', async () => {
        btnPreprocessDataset.textContent = 'Loading catalog…';
        btnPreprocessDataset.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/dataset/preprocess`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_type: selectDatasetType.value })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                if (data.metadata) applyCatalogMetadata(data.metadata, { browsable: false });
                const statusEl = document.getElementById('source-browse-status');
                if (statusEl) {
                    statusEl.textContent = data.message || 'Catalog metadata applied (not disk browse).';
                }
                pipelineWizardModal.classList.remove('hidden');
            }
        } catch (e) {
            console.error(e);
            const statusEl = document.getElementById('source-browse-status');
            if (statusEl) statusEl.textContent = 'Catalog metadata load failed.';
            notify('Catalog metadata load failed.', 'error');
        } finally {
            btnPreprocessDataset.textContent = 'Load Catalog Metadata';
            btnPreprocessDataset.disabled = false;
        }
    });

    const btnBrowseSource = document.getElementById('btn-browse-source');
    if (btnBrowseSource) {
        btnBrowseSource.addEventListener('click', async () => {
            const sourcePath = datasetSourcePath?.value || document.getElementById('input-source')?.value || 'data';
            const statusEl = document.getElementById('source-browse-status');
            const gallery = document.getElementById('source-browse-gallery');
            btnBrowseSource.disabled = true;
            btnBrowseSource.textContent = 'Scanning…';
            if (statusEl) statusEl.textContent = `Validating ${sourcePath}…`;
            try {
                const res = await fetch(`${API_BASE}/api/dataset/browse?source_path=${encodeURIComponent(sourcePath)}&limit=48`);
                const data = await res.json();
                if (gallery) {
                    gallery.innerHTML = '';
                    (data.images || []).forEach((img) => {
                        const el = document.createElement('img');
                        el.className = 'thumb';
                        el.src = img.preview_url;
                        el.title = img.path;
                        el.alt = img.name;
                        gallery.appendChild(el);
                    });
                }
                if (statusEl) {
                    statusEl.textContent = data.browsable
                        ? `Browsable: ${data.count} image(s) under ${data.source_path}`
                        : (data.empty_reason || 'Nothing browsable at this path.');
                }
                const badge = document.getElementById('ds-meta-badge');
                if (badge) {
                    if (data.browsable) {
                        badge.textContent = `${data.count} images browsable on disk`;
                        badge.style.background = 'rgba(0, 255, 170, 0.15)';
                        badge.style.color = '#00ffaa';
                        badge.style.border = '1px solid rgba(0, 255, 170, 0.3)';
                        document.getElementById('ds-kpi-loaded-rows').textContent = data.count;
                    } else {
                        badge.textContent = 'Not browsable — path empty or missing';
                        badge.style.background = 'rgba(255, 68, 68, 0.12)';
                        badge.style.color = '#ff8888';
                        badge.style.border = '1px solid rgba(255, 68, 68, 0.35)';
                    }
                }
            } catch (e) {
                if (statusEl) statusEl.textContent = `Browse failed: ${e.message}`;
            } finally {
                btnBrowseSource.disabled = false;
                btnBrowseSource.textContent = 'Validate & Browse Images Path';
            }
        });
    }

    loadCatalogMetadata(selectDatasetType?.value || 'local');

    btnWizardSubmit.addEventListener('click', async () => {
        const annotationTool = document.getElementById('wizard-annotation-tool').value;
        const trainingFramework = document.getElementById('wizard-training-framework').value;
        const validationMethod = document.getElementById('wizard-validation-method').value;

        btnWizardSubmit.textContent = 'Saving Settings...';
        btnWizardSubmit.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/dataset/save-pipeline-tools`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    annotation_tool: annotationTool,
                    training_framework: trainingFramework,
                    validation_method: validationMethod
                })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                notify(
                    `Pipeline configuration saved\nAnnotation: ${annotationTool}\nTraining: ${trainingFramework}\nValidation: ${validationMethod}`,
                    'success'
                );
                pipelineWizardModal.classList.add('hidden');
            }
        } catch (e) {
            console.error(e);
            notify('Failed to save pipeline configuration settings.', 'error');
        } finally {
            btnWizardSubmit.textContent = 'Save Pipeline Configuration';
            btnWizardSubmit.disabled = false;
        }
    });

    renderSourceLinks(selectDatasetType.value);

    // Camera view switching
    const cameraButtons = document.querySelectorAll('[data-view]');
    cameraButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            cameraButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const view = btn.dataset.view;
            if (currentMitlData && currentMitlData.views) {
                mitlCameraImg.src = currentMitlData.views[view];
                mitlCameraLabel.textContent = `${view.toUpperCase()}_CAM`;
            }
        });
    });

    // Render current MITL telemetry and boxes
    function renderMitlData() {
        if (!currentMitlData) return;

        // Telemetry
        const t = currentMitlData.telemetry;
        mitlGps.textContent = `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}`;
        mitlSpeed.textContent = `${formatNumber(t.speed_kmh)} km/h`;
        mitlAccel.textContent = `${formatNumber(t.accel_mps2)} m/s²`;
        mitlImu.textContent = `Pitch: ${formatNumber(t.imu_pitch)}, Roll: ${formatNumber(t.imu_roll)}`;

        // View image
        const activeBtn = document.querySelector('[data-view].active') || cameraButtons[0];
        const activeView = activeBtn.dataset.view;
        mitlCameraImg.src = currentMitlData.views[activeView];
        mitlCameraLabel.textContent = `${activeView.toUpperCase()}_CAM`;

        // Reasoning Trace
        mitlCocTrace.value = currentMitlData.coc_trace;

        // Bounding Boxes Table
        renderBoxesTable();
    }

    const TAXONOMY_CLASSES = [
        'car', 'suv', 'van', 'pickup', 'taxi',
        'bus', 'truck', 'semi-trailer', 'emergency-vehicle',
        'pedestrian', 'child', 'wheelchair-user', 'cyclist', 'e-bike',
        'animal'
    ];

    function renderBoxesTable() {
        mitlBoxRows.innerHTML = '';
        if (!currentMitlData || !currentMitlData.annotations) return;

        currentMitlData.annotations.forEach((ann, idx) => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border)';

            let optionsHtml = '';
            TAXONOMY_CLASSES.forEach(cls => {
                const selectedAttr = cls === ann.label ? 'selected' : '';
                optionsHtml += `<option value="${cls}" ${selectedAttr}>${cls}</option>`;
            });

            tr.innerHTML = `
                <td style="padding: 8px;">${ann.id}</td>
                <td style="padding: 8px;">
                    <select class="mitl-input-class font-mono" style="width: 120px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 4px; color: #fff; padding: 2px 4px;">
                        ${optionsHtml}
                    </select>
                </td>
                <td style="padding: 8px;">
                    <input type="text" class="mitl-input-box font-mono" value="[${ann.box.join(', ')}]" style="width: 160px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 4px; color: #fff; padding: 2px 4px;">
                </td>
                <td style="padding: 8px;">
                    <input type="number" step="0.01" class="mitl-input-conf font-mono" value="${ann.conf}" style="width: 60px; background: rgba(0,0,0,0.3); border: 1px solid var(--border); border-radius: 4px; color: #fff; padding: 2px 4px;">
                </td>
                <td style="padding: 8px; text-align: right;">
                    <button class="btn btn-danger btn-small btn-mitl-delete" data-idx="${idx}" style="padding: 2px 8px;">Delete</button>
                </td>
            `;
            mitlBoxRows.appendChild(tr);
        });

        // Add Delete Listeners
        const deleteButtons = mitlBoxRows.querySelectorAll('.btn-mitl-delete');
        deleteButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(btn.dataset.idx);
                currentMitlData.annotations.splice(idx, 1);
                renderBoxesTable();
            });
        });
    }

    // Add Box Handler
    btnMitlAddBox.addEventListener('click', () => {
        if (!currentMitlData) {
            notify('Please load a dataset first.', 'warning');
            return;
        }
        const nextId = currentMitlData.annotations.length > 0 
            ? Math.max(...currentMitlData.annotations.map(a => a.id)) + 1 
            : 1;
            
        currentMitlData.annotations.push({
            id: nextId,
            label: 'car',
            box: [100, 100, 50, 50],
            conf: 0.90
        });
        renderBoxesTable();
    });

    // Save Human Edits
    btnMitlSave.addEventListener('click', async () => {
        if (!currentMitlData) {
            notify('No dataset currently loaded to save.', 'warning');
            return;
        }

        // Collect inputs from DOM
        const rows = mitlBoxRows.querySelectorAll('tr');
        const updatedAnns = [];
        let parseError = false;

        rows.forEach((row, i) => {
            const id = parseInt(row.cells[0].textContent);
            const label = row.querySelector('.mitl-input-class').value.trim();
            const boxStr = row.querySelector('.mitl-input-box').value;
            const conf = parseFloat(row.querySelector('.mitl-input-conf').value);

            try {
                const box = JSON.parse(boxStr);
                if (!Array.isArray(box) || box.length !== 4) {
                    throw new Error();
                }
                updatedAnns.push({ id, label, box, conf });
            } catch (e) {
                parseError = true;
                row.querySelector('.mitl-input-box').style.borderColor = 'var(--danger)';
            }
        });

        if (parseError) {
            notify('Invalid bounding box format. Use coordinates [x, y, w, h].', 'error');
            return;
        }

        // Save
        try {
            const res = await fetch(`${API_BASE}/api/mitl/annotations`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_type: selectDatasetType.value,
                    annotations: updatedAnns,
                    coc_trace: mitlCocTrace.value
                })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                // Update local memory
                currentMitlData.annotations = updatedAnns;
                currentMitlData.coc_trace = mitlCocTrace.value;
                notify('Triage results saved to runs/mitl_annotations.json.', 'success');
            }
        } catch (e) {
            console.error(e);
            notify('Failed to save human edits.', 'error');
        }
    });

    // Ask Gemma 4 critique
    btnMitlAudit.addEventListener('click', async () => {
        if (!currentMitlData) {
            notify('No dataset currently loaded to audit.', 'warning');
            return;
        }

        btnMitlAudit.textContent = 'Auditing (Gemma 4)...';
        btnMitlAudit.disabled = true;
        mitlAuditResults.classList.remove('hidden');
        mitlAuditText.innerHTML = '<span class="copilot-spinner" style="display:inline-block; vertical-align:middle; margin-right:8px;"></span> Evaluating human adjustments against causation model...';

        try {
            const res = await fetch(`${API_BASE}/api/mitl/evaluate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    annotations: currentMitlData.annotations,
                    coc_trace: mitlCocTrace.value,
                    original_coc: originalCocText
                })
            });
            const data = await res.json();
            mitlAuditText.innerHTML = parseMarkdown(data.critique);
        } catch (e) {
            console.error(e);
            mitlAuditText.textContent = 'Failed to connect to triage auditor.';
        } finally {
            btnMitlAudit.textContent = 'Ask Gemma 4 Critique';
            btnMitlAudit.disabled = false;
        }
    });

    // --- MODEL BENCHMARKING CONTROLLERS ---
    const btnRunBenchmark = document.getElementById('btn-run-benchmark');
    const selectModelA = document.getElementById('benchmark-model-a');
    const selectModelB = document.getElementById('benchmark-model-b');
    const benchmarkTableRows = document.getElementById('benchmark-table-rows');
    const benchmarkLblA = document.getElementById('benchmark-lbl-a');
    const benchmarkLblB = document.getElementById('benchmark-lbl-b');

    let benchmarkCache = null;

    async function loadBenchmarkData() {
        try {
            const res = await fetch(`${API_BASE}/api/benchmark/compare?dataset=${selectDatasetType.value}`);
            const data = await res.json();
            if (data.status === 'ok') {
                benchmarkCache = data.benchmarks;
                renderBenchmarkTable();
                drawBenchmarkChart();
            }
        } catch (e) {
            console.error(e);
        }
    }

    function renderBenchmarkTable() {
        if (!benchmarkCache) return;
        benchmarkTableRows.innerHTML = '';
        Object.entries(benchmarkCache).forEach(([key, model]) => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid var(--border)';
            tr.innerHTML = `
                <td style="padding: 10px; font-weight: 500; color: #fff;">${model.name}</td>
                <td style="padding: 10px;">${model.type}</td>
                <td style="padding: 10px; font-family: var(--font-mono);">${formatNumber(model.latency_ms)} ms</td>
                <td style="padding: 10px; font-family: var(--font-mono);">${formatPercent(model.map50)}</td>
                <td style="padding: 10px; font-family: var(--font-mono);">${formatPercent(model.risk_weighted_recall)}</td>
                <td style="padding: 10px; font-family: var(--font-mono);">${formatPercent(model.recall_critical_distance)}</td>
                <td style="padding: 10px; font-family: var(--font-mono);">${formatPercent(model.vru_recall)}</td>
                <td style="padding: 10px; font-size: 11px;">${model.coc_support}</td>
            `;
            benchmarkTableRows.appendChild(tr);
        });
    }

    function drawBenchmarkChart() {
        const canvas = document.getElementById('benchmark-comparison-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const modelAKey = selectModelA.value;
        const modelBKey = selectModelB.value;

        if (!benchmarkCache || !benchmarkCache[modelAKey] || !benchmarkCache[modelBKey]) return;

        const modelA = benchmarkCache[modelAKey];
        const modelB = benchmarkCache[modelBKey];

        // Labels
        benchmarkLblA.textContent = modelA.name;
        benchmarkLblB.textContent = modelB.name;

        // Background
        ctx.fillStyle = '#04060b';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Chart metrics config
        const metrics = [
            { label: 'mAP [50]', valA: modelA.map50, valB: modelB.map50 },
            { label: 'Risk-Wt Recall', valA: modelA.risk_weighted_recall, valB: modelB.risk_weighted_recall },
            { label: 'Recall @ Crit Dist', valA: modelA.recall_critical_distance, valB: modelB.recall_critical_distance },
            { label: 'VRU Recall', valA: modelA.vru_recall, valB: modelB.vru_recall }
        ];

        const barWidth = 20;
        const groupSpacing = 35;
        const startX = 50;
        const startY = 30;
        const chartHeight = 130;

        // Draw axes and grids
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = startY + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(canvas.width - 20, y);
            ctx.stroke();

            // Y percentage label
            ctx.fillStyle = '#6b7280';
            ctx.font = '10px sans-serif';
            ctx.fillText(`${(100 - i * 25)}%`, startX - 30, y + 4);
        }

        // Draw bars
        metrics.forEach((m, idx) => {
            const x = startX + 30 + idx * (barWidth * 2 + groupSpacing);

            // Bar A (Model A) - Blue
            const hA = m.valA * chartHeight;
            const yA = startY + chartHeight - hA;
            ctx.fillStyle = '#3b82f6';
            ctx.fillRect(x, yA, barWidth, hA);

            // Bar B (Model B) - Purple
            const hB = m.valB * chartHeight;
            const yB = startY + chartHeight - hB;
            ctx.fillStyle = '#a855f7';
            ctx.fillRect(x + barWidth + 4, yB, barWidth, hB);

            // X label
            ctx.fillStyle = '#9ca3af';
            ctx.font = '9px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(m.label, x + barWidth, startY + chartHeight + 15);
        });

        // Draw Legend
        ctx.textAlign = 'left';
        ctx.font = '10px sans-serif';
        // Legend A
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(startX, startY + chartHeight + 28, 12, 12);
        ctx.fillStyle = '#d1d5db';
        ctx.fillText(modelA.name, startX + 18, startY + chartHeight + 38);

        // Legend B
        ctx.fillStyle = '#a855f7';
        ctx.fillRect(startX + 160, startY + chartHeight + 28, 12, 12);
        ctx.fillStyle = '#d1d5db';
        ctx.fillText(modelB.name, startX + 178, startY + chartHeight + 38);
    }

    btnRunBenchmark.addEventListener('click', loadBenchmarkData);
    selectModelA.addEventListener('change', drawBenchmarkChart);
    selectModelB.addEventListener('change', drawBenchmarkChart);

    // Fetch benchmarks when active nav tab changes
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.stage === 'benchmark') {
                loadBenchmarkData();
            }
        });
    });
    nextBtn.addEventListener('click', () => {
        if (nextBtn.dataset.next === 'benchmark') {
            loadBenchmarkData();
        }
    });

    // --- MCP CONFIGURATION & TOGGLE CONTROLLERS ---
    const mcpServersList = document.getElementById('mcp-servers-list');
    const mcpRawTextarea = document.getElementById('mcp-raw-textarea');
    const btnMcpRefresh = document.getElementById('btn-mcp-refresh');
    const btnMcpSave = document.getElementById('btn-mcp-save');
    const mcpSaveStatus = document.getElementById('mcp-save-status');

    async function loadMcpConfig() {
        try {
            const res = await fetch(`${API_BASE}/api/mcp/config`);
            const data = await res.json();
            if (data.status === 'ok') {
                mcpRawTextarea.value = data.raw;
                renderMcpServers(data.config);
            }
        } catch (e) {
            console.error(e);
            mcpRawTextarea.value = "Failed to load MCP configuration from backend.";
        }
    }

    function renderMcpServers(config) {
        mcpServersList.innerHTML = '';
        const servers = config.mcpServers || {};
        
        // We find all servers (including disabled ones with -disabled suffix)
        const normalized = {};
        Object.entries(servers).forEach(([key, value]) => {
            const isActive = !key.endsWith('-disabled');
            const cleanKey = key.replace('-disabled', '');
            normalized[cleanKey] = {
                active: isActive,
                config: value,
                originalKey: key
            };
        });

        if (Object.keys(normalized).length === 0) {
            mcpServersList.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 20px;">No MCP Servers configured in this file.</div>';
            return;
        }

        Object.entries(normalized).forEach(([name, info]) => {
            const card = document.createElement('div');
            card.style.background = 'rgba(255, 255, 255, 0.02)';
            card.style.border = '1px solid var(--border)';
            card.style.borderRadius = '8px';
            card.style.padding = '14px';
            card.style.display = 'flex';
            card.style.alignItems = 'center';
            card.style.justifyContent = 'space-between';

            const activeLabel = info.active ? '<span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #10b981;">Active</span>' : '<span class="badge" style="background: rgba(239, 68, 68, 0.15); color: #ef4444;">Disabled</span>';

            card.innerHTML = `
                <div>
                    <h4 style="font-size: 14px; font-weight: 600; color: #fff; display: flex; align-items: center; gap: 8px;">
                        ${name} ${activeLabel}
                    </h4>
                    <p style="font-size: 11px; color: var(--text-muted); margin-top: 4px; font-family: var(--font-mono);">
                        cmd: ${info.config.command} ${info.config.args ? info.config.args.join(' ') : ''}
                    </p>
                </div>
                <div>
                    <label class="switch-container" style="position: relative; display: inline-block; width: 44px; height: 22px;">
                        <input type="checkbox" class="mcp-toggle-checkbox" data-server="${name}" ${info.active ? 'checked' : ''} style="opacity: 0; width: 0; height: 0;">
                        <span class="slider" style="position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #374151; transition: .3s; border-radius: 34px;"></span>
                    </label>
                </div>
            `;

            // Style individual slider switches
            const slider = card.querySelector('.slider');
            const checkbox = card.querySelector('input');
            const updateSliderStyle = () => {
                if (checkbox.checked) {
                    slider.style.backgroundColor = 'var(--primary)';
                    slider.style.boxShadow = '0 0 8px var(--primary-glow)';
                } else {
                    slider.style.backgroundColor = '#374151';
                    slider.style.boxShadow = 'none';
                }
            };
            
            // Add custom dynamic bullet position
            slider.innerHTML = `<span style="position: absolute; content: ''; height: 16px; width: 16px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; transform: ${checkbox.checked ? 'translateX(22px)' : 'none'};"></span>`;
            
            checkbox.addEventListener('change', async () => {
                const bullet = slider.querySelector('span');
                bullet.style.transform = checkbox.checked ? 'translateX(22px)' : 'none';
                updateSliderStyle();
                
                try {
                    const res = await fetch(`${API_BASE}/api/mcp/toggle`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ server_name: name, active: checkbox.checked })
                    });
                    const resData = await res.json();
                    if (resData.status === 'ok' || resData.status === 'warning') {
                        loadMcpConfig(); // Refresh raw text and state
                    }
                } catch (e) {
                    console.error(e);
                    notify('Failed to toggle MCP server state.', 'error');
                    checkbox.checked = !checkbox.checked;
                    bullet.style.transform = checkbox.checked ? 'translateX(22px)' : 'none';
                    updateSliderStyle();
                }
            });

            updateSliderStyle();
            mcpServersList.appendChild(card);
        });
    }

    btnMcpRefresh.addEventListener('click', loadMcpConfig);

    btnMcpSave.addEventListener('click', async () => {
        btnMcpSave.textContent = 'Saving...';
        btnMcpSave.disabled = true;
        mcpSaveStatus.classList.add('hidden');

        try {
            const res = await fetch(`${API_BASE}/api/mcp/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config_json: mcpRawTextarea.value })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                mcpSaveStatus.textContent = 'Configuration saved and updated successfully.';
                mcpSaveStatus.className = 'status-msg success-msg';
                mcpSaveStatus.classList.remove('hidden');
                loadMcpConfig();
            }
        } catch (e) {
            console.error(e);
            mcpSaveStatus.textContent = 'Error writing raw JSON config file.';
            mcpSaveStatus.className = 'status-msg error-msg';
            mcpSaveStatus.classList.remove('hidden');
        } finally {
            btnMcpSave.textContent = 'Save Raw JSON';
            btnMcpSave.disabled = false;
        }
    });

    // --- SSAM STATEWIDE GIS DASHBOARD CONTROLLERS ---
    const ssamGridBody = document.getElementById('ssam-grid-body');
    const ssamPageInfo = document.getElementById('ssam-page-info');
    const ssamPagePrev = document.getElementById('ssam-page-prev');
    const ssamPageNext = document.getElementById('ssam-page-next');
    const ssamPageSize = document.getElementById('ssam-page-size');
    const ssamApplyFilters = document.getElementById('ssam-apply-filters');
    const ssamResetFilters = document.getElementById('ssam-reset-filters');
    const ssamFilterSearch = document.getElementById('ssam-filter-search');
    const ssamFilterCounty = document.getElementById('ssam-filter-county');
    const ssamFilterTtc = document.getElementById('ssam-filter-ttc');
    const ssamTtcVal = document.getElementById('ssam-ttc-val');
    const ssamMapCanvas = document.getElementById('ssam-map-canvas');
    const ssamMapTooltip = document.getElementById('ssam-map-tooltip');
    const ssamDrawer = document.getElementById('ssam-severity-drawer');
    const ssamDrawerClose = document.getElementById('ssam-drawer-close');
    const ssamDrawerTitle = document.getElementById('ssam-drawer-title');
    const ssamDrawerMetrics = document.getElementById('ssam-drawer-metrics');
    const ssamDrawerAnnotation = document.getElementById('ssam-drawer-annotation');
    const ssamDrawerSave = document.getElementById('ssam-drawer-save');

    let ssamState = {
        page: 1,
        pageSize: 25,
        sortBy: 'severity_index',
        sortDir: 'desc',
        rows: [],
        geojson: null,
        total: 0,
        totalPages: 1,
        summary: {},
        selectedRow: null,
        mapZoom: 1,
        mapOffsetX: 0,
        mapOffsetY: 0,
    };

    // California bounding box
    const CA_BOUNDS = { latMin: 32.4, latMax: 42.1, lngMin: -124.5, lngMax: -114.0 };

    function ssamLatLngToCanvas(lat, lng, canvas) {
        const w = canvas.width;
        const h = canvas.height;
        const padX = 40, padY = 30;
        const drawW = (w - padX * 2) * ssamState.mapZoom;
        const drawH = (h - padY * 2) * ssamState.mapZoom;
        const cx = (w / 2) + ssamState.mapOffsetX;
        const cy = (h / 2) + ssamState.mapOffsetY;

        const xNorm = (lng - CA_BOUNDS.lngMin) / (CA_BOUNDS.lngMax - CA_BOUNDS.lngMin);
        const yNorm = 1 - (lat - CA_BOUNDS.latMin) / (CA_BOUNDS.latMax - CA_BOUNDS.latMin);

        const x = cx - drawW / 2 + xNorm * drawW;
        const y = cy - drawH / 2 + yNorm * drawH;
        return { x, y };
    }

    function ssamSeverityColor(label) {
        switch (label) {
            case 'Critical': return { fill: '#ef4444', glow: 'rgba(239,68,68,0.4)' };
            case 'High': return { fill: '#f59e0b', glow: 'rgba(245,158,11,0.4)' };
            case 'Medium': return { fill: '#8b5cf6', glow: 'rgba(139,92,246,0.4)' };
            default: return { fill: '#10b981', glow: 'rgba(16,185,129,0.4)' };
        }
    }

    function ssamDrawMap() {
        const canvas = ssamMapCanvas;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;

        // Size canvas to container
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.scale(dpr, dpr);
        canvas.width = rect.width;
        canvas.height = rect.height;

        const w = canvas.width;
        const h = canvas.height;

        // Background
        ctx.fillStyle = '#060a14';
        ctx.fillRect(0, 0, w, h);

        // Draw a simplified California outline (convex hull approximation)
        const caOutline = [
            [42.0, -124.2], [41.99, -120.0], [39.0, -120.0], [38.5, -119.9],
            [36.0, -117.6], [35.0, -115.7], [34.5, -114.6], [32.7, -114.7],
            [32.5, -117.1], [33.0, -117.3], [33.5, -117.8], [33.9, -118.4],
            [34.0, -118.5], [34.5, -120.5], [35.3, -120.9], [36.3, -121.9],
            [37.0, -122.4], [37.8, -122.5], [38.3, -123.0], [39.0, -123.7],
            [40.4, -124.3], [42.0, -124.2],
        ];

        ctx.beginPath();
        caOutline.forEach((pt, i) => {
            const { x, y } = ssamLatLngToCanvas(pt[0], pt[1], canvas);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(20, 30, 55, 0.5)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.25)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.03)';
        ctx.lineWidth = 0.5;
        for (let lat = 33; lat <= 42; lat++) {
            const p1 = ssamLatLngToCanvas(lat, CA_BOUNDS.lngMin, canvas);
            const p2 = ssamLatLngToCanvas(lat, CA_BOUNDS.lngMax, canvas);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }
        for (let lng = -124; lng <= -114; lng += 2) {
            const p1 = ssamLatLngToCanvas(CA_BOUNDS.latMin, lng, canvas);
            const p2 = ssamLatLngToCanvas(CA_BOUNDS.latMax, lng, canvas);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        }

        // Draw data points
        if (!ssamState.geojson) return;
        const features = ssamState.geojson.features || [];

        // Draw glow pass first
        features.forEach(f => {
            const [lng, lat] = f.geometry.coordinates;
            const { x, y } = ssamLatLngToCanvas(lat, lng, canvas);
            const sev = f.properties.severity_label;
            const colors = ssamSeverityColor(sev);
            const r = sev === 'Critical' ? 8 : sev === 'High' ? 6 : 5;
            ctx.beginPath();
            ctx.arc(x, y, r + 4, 0, Math.PI * 2);
            ctx.fillStyle = colors.glow;
            ctx.fill();
        });

        // Solid dots
        features.forEach(f => {
            const [lng, lat] = f.geometry.coordinates;
            const { x, y } = ssamLatLngToCanvas(lat, lng, canvas);
            const sev = f.properties.severity_label;
            const colors = ssamSeverityColor(sev);
            const r = sev === 'Critical' ? 6 : sev === 'High' ? 5 : 4;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, Math.PI * 2);
            ctx.fillStyle = colors.fill;
            ctx.fill();
            ctx.strokeStyle = 'rgba(0,0,0,0.4)';
            ctx.lineWidth = 1;
            ctx.stroke();
        });

        // City labels
        const cities = [
            { name: 'San Francisco', lat: 37.78, lng: -122.42 },
            { name: 'Los Angeles', lat: 34.05, lng: -118.24 },
            { name: 'San Diego', lat: 32.72, lng: -117.16 },
            { name: 'Sacramento', lat: 38.58, lng: -121.49 },
            { name: 'Fresno', lat: 36.74, lng: -119.77 },
            { name: 'Bakersfield', lat: 35.37, lng: -119.02 },
            { name: 'Redding', lat: 40.59, lng: -122.39 },
            { name: 'Eureka', lat: 40.80, lng: -124.16 },
        ];
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'left';
        cities.forEach(c => {
            const { x, y } = ssamLatLngToCanvas(c.lat, c.lng, canvas);
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.fillText(c.name, x + 10, y + 4);
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.fill();
        });
    }

    // Map tooltip on hover
    ssamMapCanvas?.addEventListener('mousemove', (e) => {
        if (!ssamState.geojson) return;
        const rect = ssamMapCanvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let hit = null;
        const features = ssamState.geojson.features || [];
        for (const f of features) {
            const [lng, lat] = f.geometry.coordinates;
            const { x, y } = ssamLatLngToCanvas(lat, lng, ssamMapCanvas);
            const dist = Math.sqrt((mx - x) ** 2 + (my - y) ** 2);
            if (dist < 12) { hit = f.properties; break; }
        }

        if (hit) {
            ssamMapTooltip.classList.remove('hidden');
            ssamMapTooltip.innerHTML = `
                <strong>${hit.street_name}</strong><br>
                <span style="color:var(--text-muted)">${hit.county} County</span><br>
                Type: ${hit.conflict_type} · TTC: ${formatNumber(hit.min_ttc)}s<br>
                Severity: <span style="color:${ssamSeverityColor(hit.severity_label).fill}; font-weight:600;">${hit.severity_label} (${formatPercent(hit.severity_index)})</span>
            `;
            let tx = mx + 16, ty = my - 10;
            if (tx + 240 > rect.width) tx = mx - 250;
            if (ty < 0) ty = 10;
            ssamMapTooltip.style.left = tx + 'px';
            ssamMapTooltip.style.top = ty + 'px';
        } else {
            ssamMapTooltip.classList.add('hidden');
        }
    });

    ssamMapCanvas?.addEventListener('mouseleave', () => {
        ssamMapTooltip.classList.add('hidden');
    });

    // Map zoom / pan controls
    document.getElementById('ssam-map-zoom-in')?.addEventListener('click', () => {
        ssamState.mapZoom = Math.min(4, ssamState.mapZoom * 1.3);
        ssamDrawMap();
    });
    document.getElementById('ssam-map-zoom-out')?.addEventListener('click', () => {
        ssamState.mapZoom = Math.max(0.5, ssamState.mapZoom / 1.3);
        ssamDrawMap();
    });
    document.getElementById('ssam-map-reset')?.addEventListener('click', () => {
        ssamState.mapZoom = 1;
        ssamState.mapOffsetX = 0;
        ssamState.mapOffsetY = 0;
        ssamDrawMap();
    });

    // Map drag panning
    let ssamMapDrag = false, ssamMapLastX = 0, ssamMapLastY = 0;
    ssamMapCanvas?.addEventListener('mousedown', (e) => {
        ssamMapDrag = true;
        ssamMapLastX = e.clientX;
        ssamMapLastY = e.clientY;
        ssamMapCanvas.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (!ssamMapDrag) return;
        ssamState.mapOffsetX += e.clientX - ssamMapLastX;
        ssamState.mapOffsetY += e.clientY - ssamMapLastY;
        ssamMapLastX = e.clientX;
        ssamMapLastY = e.clientY;
        ssamDrawMap();
    });
    window.addEventListener('mouseup', () => {
        ssamMapDrag = false;
        if (ssamMapCanvas) ssamMapCanvas.style.cursor = 'crosshair';
    });

    // Mouse wheel zoom
    ssamMapCanvas?.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        ssamState.mapZoom = Math.max(0.5, Math.min(4, ssamState.mapZoom * delta));
        ssamDrawMap();
    }, { passive: false });

    // TTC slider
    ssamFilterTtc?.addEventListener('input', () => {
        ssamTtcVal.textContent = formatNumber(parseFloat(ssamFilterTtc.value));
    });

    // Build filter query
    function ssamBuildQuery() {
        const q = {
            page: ssamState.page,
            page_size: ssamState.pageSize,
            sort_by: ssamState.sortBy,
            sort_dir: ssamState.sortDir,
        };

        const search = ssamFilterSearch?.value?.trim();
        if (search) q.search = search;

        const counties = Array.from(ssamFilterCounty?.selectedOptions || []).map(o => o.value);
        if (counties.length > 0) q.counties = counties;

        const conflictChecks = document.querySelectorAll('#ssam-filter-conflict-types input:checked');
        const conflictTypes = Array.from(conflictChecks).map(c => c.value);
        if (conflictTypes.length > 0 && conflictTypes.length < 3) q.conflict_types = conflictTypes;

        const sevChecks = document.querySelectorAll('#ssam-filter-severity input:checked');
        const sevLabels = Array.from(sevChecks).map(c => c.value);
        if (sevLabels.length > 0 && sevLabels.length < 4) q.severity_labels = sevLabels;

        const ttcMax = parseFloat(ssamFilterTtc?.value || '1.5');
        if (ttcMax < 1.5) q.ttc_max = ttcMax;

        return q;
    }

    // Animated KPI counters
    function ssamAnimateCounter(el, target) {
        const duration = 600;
        const start = parseInt(el.textContent) || 0;
        const range = target - start;
        const startTime = performance.now();
        function step(t) {
            const progress = Math.min((t - startTime) / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3);
            el.textContent = Math.round(start + range * ease);
            if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
    }

    // Load statewide data
    async function ssamLoadStatewide() {
        const q = ssamBuildQuery();
        try {
            const res = await fetch(`${API_BASE}/api/ssam/statewide`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(q),
            });
            const data = await res.json();
            if (data.status !== 'ok') return;

            ssamState.rows = data.rows;
            ssamState.geojson = data.geojson;
            ssamState.total = data.total;
            ssamState.page = data.page;
            ssamState.totalPages = data.total_pages;
            ssamState.summary = data.summary;

            // Update KPIs
            ssamAnimateCounter(document.getElementById('ssam-kpi-total'), data.total);
            ssamAnimateCounter(document.getElementById('ssam-kpi-critical'), data.summary.Critical || 0);
            ssamAnimateCounter(document.getElementById('ssam-kpi-high'), data.summary.High || 0);
            ssamAnimateCounter(document.getElementById('ssam-kpi-medium'), data.summary.Medium || 0);
            ssamAnimateCounter(document.getElementById('ssam-kpi-low'), data.summary.Low || 0);

            // Populate county filter (first load only)
            if (ssamFilterCounty && ssamFilterCounty.options.length === 0 && data.filter_options) {
                data.filter_options.counties.forEach(c => {
                    const opt = document.createElement('option');
                    opt.value = c; opt.textContent = c;
                    ssamFilterCounty.appendChild(opt);
                });
            }

            ssamRenderGrid();
            ssamDrawMap();

            // Update pagination
            ssamPageInfo.textContent = `Page ${data.page} / ${data.total_pages}  (${data.total} records)`;
            ssamPagePrev.disabled = data.page <= 1;
            ssamPageNext.disabled = data.page >= data.total_pages;
        } catch (e) {
            console.error('SSAM statewide fetch error:', e);
        }
    }

    function ssamRenderGrid() {
        ssamGridBody.innerHTML = '';
        ssamState.rows.forEach(row => {
            const tr = document.createElement('tr');
            const sevClass = `severity-${row.severity_label.toLowerCase()}`;
            tr.innerHTML = `
                <td class="street-name-cell">${row.street_name}</td>
                <td>${row.county}</td>
                <td>${row.conflict_type}</td>
                <td class="mono-cell">${formatNumber(row.min_ttc)}s</td>
                <td class="mono-cell">${formatNumber(row.min_pet)}s</td>
                <td class="mono-cell">${formatNumber(row.max_speed)} m/s</td>
                <td><span class="ssam-severity-badge ${sevClass}">${row.severity_label} ${formatPercent(row.severity_index)}</span></td>
            `;
            tr.addEventListener('click', () => ssamOpenDrawer(row));
            ssamGridBody.appendChild(tr);
        });
    }

    // Sorting
    document.querySelectorAll('.ssam-data-table th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (ssamState.sortBy === col) {
                ssamState.sortDir = ssamState.sortDir === 'desc' ? 'asc' : 'desc';
            } else {
                ssamState.sortBy = col;
                ssamState.sortDir = 'desc';
            }
            // Update active indicator
            document.querySelectorAll('.ssam-data-table th').forEach(t => t.classList.remove('sort-active'));
            th.classList.add('sort-active');
            ssamState.page = 1;
            ssamLoadStatewide();
        });
    });

    // Pagination
    ssamPagePrev?.addEventListener('click', () => {
        if (ssamState.page > 1) { ssamState.page--; ssamLoadStatewide(); }
    });
    ssamPageNext?.addEventListener('click', () => {
        if (ssamState.page < ssamState.totalPages) { ssamState.page++; ssamLoadStatewide(); }
    });
    ssamPageSize?.addEventListener('change', () => {
        ssamState.pageSize = parseInt(ssamPageSize.value);
        ssamState.page = 1;
        ssamLoadStatewide();
    });

    // Filters
    ssamApplyFilters?.addEventListener('click', () => {
        ssamState.page = 1;
        ssamLoadStatewide();
    });
    ssamResetFilters?.addEventListener('click', () => {
        ssamFilterSearch.value = '';
        Array.from(ssamFilterCounty.options).forEach(o => o.selected = false);
        document.querySelectorAll('#ssam-filter-conflict-types input').forEach(c => c.checked = true);
        document.querySelectorAll('#ssam-filter-severity input').forEach(c => c.checked = true);
        ssamFilterTtc.value = '1.5';
        ssamTtcVal.textContent = '1.5';
        ssamState.page = 1;
        ssamState.sortBy = 'severity_index';
        ssamState.sortDir = 'desc';
        ssamState.mapZoom = 1;
        ssamState.mapOffsetX = 0;
        ssamState.mapOffsetY = 0;
        ssamLoadStatewide();
    });

    // Enter key in search
    ssamFilterSearch?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { ssamState.page = 1; ssamLoadStatewide(); }
    });

    // Severity Drawer
    function ssamOpenDrawer(row) {
        ssamState.selectedRow = row;
        ssamDrawerTitle.textContent = row.street_name;
        ssamDrawerAnnotation.value = row.manual_annotation || '';
        const sevClass = `severity-${row.severity_label.toLowerCase()}`;
        ssamDrawerMetrics.innerHTML = `
            <div class="ssam-drawer-metric"><span class="metric-val">${formatNumber(row.min_ttc)}s</span><span class="metric-lbl">Min TTC</span></div>
            <div class="ssam-drawer-metric"><span class="metric-val">${formatNumber(row.min_pet)}s</span><span class="metric-lbl">Min PET</span></div>
            <div class="ssam-drawer-metric"><span class="metric-val">${formatNumber(row.max_speed)}</span><span class="metric-lbl">Max ΔSpeed (m/s)</span></div>
            <div class="ssam-drawer-metric"><span class="metric-val"><span class="ssam-severity-badge ${sevClass}">${row.severity_label}</span></span><span class="metric-lbl">Severity</span></div>
            <div class="ssam-drawer-metric"><span class="metric-val">${row.county}</span><span class="metric-lbl">County</span></div>
            <div class="ssam-drawer-metric"><span class="metric-val">${row.conflict_type}</span><span class="metric-lbl">Conflict Type</span></div>
        `;
        ssamDrawer.classList.remove('hidden');
    }

    ssamDrawerClose?.addEventListener('click', () => {
        ssamDrawer.classList.add('hidden');
    });

    ssamDrawerSave?.addEventListener('click', async () => {
        if (!ssamState.selectedRow) return;
        ssamDrawerSave.textContent = 'Saving…';
        ssamDrawerSave.disabled = true;
        try {
            await fetch(`${API_BASE}/api/ssam/annotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    street_name: ssamState.selectedRow.street_name,
                    manual_annotation: ssamDrawerAnnotation.value,
                }),
            });
            ssamState.selectedRow.manual_annotation = ssamDrawerAnnotation.value;
            notify(`Annotation saved for ${ssamState.selectedRow.street_name}`, 'success');
        } catch (e) {
            notify('Failed to save annotation.', 'error');
        } finally {
            ssamDrawerSave.textContent = 'Save Annotation';
            ssamDrawerSave.disabled = false;
        }
    });

    // Fetch SSAM data when navigating to tab
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.dataset.stage === 'ssam') {
                ssamLoadStatewide();
            }
        });
    });
    nextBtn.addEventListener('click', () => {
        if (nextBtn.dataset.next === 'ssam') {
            ssamLoadStatewide();
        }
    });

    // --- DATASET YAML CONFIGURATION EDITOR ---
    const datasetYamlEditor = document.getElementById('dataset-yaml-editor');
    const btnYamlReload = document.getElementById('btn-yaml-reload');
    const btnYamlSave = document.getElementById('btn-yaml-save');
    const yamlSaveStatus = document.getElementById('yaml-save-status');

    async function loadDatasetYaml() {
        const path = dataYamlPath.value || 'coco8.yaml';
        try {
            const res = await fetch(`${API_BASE}/api/yaml/content?path=${encodeURIComponent(path)}`);
            const data = await res.json();
            if (data.status === 'ok') {
                datasetYamlEditor.value = data.content;
                yamlSaveStatus.classList.add('hidden');
            } else {
                datasetYamlEditor.value = `Error: ${data.message}`;
            }
        } catch (e) {
            console.error(e);
            datasetYamlEditor.value = 'Error: Failed to connect to server and load YAML configuration.';
        }
    }

    btnYamlReload.addEventListener('click', loadDatasetYaml);

    btnYamlSave.addEventListener('click', async () => {
        const path = dataYamlPath.value || 'coco8.yaml';
        const content = datasetYamlEditor.value;

        btnYamlSave.textContent = 'Saving...';
        btnYamlSave.disabled = true;

        try {
            const res = await fetch(`${API_BASE}/api/yaml/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path, content })
            });
            const data = await res.json();
            if (data.status === 'ok') {
                yamlSaveStatus.textContent = 'YAML config saved successfully.';
                yamlSaveStatus.className = 'status-msg success-msg';
                yamlSaveStatus.classList.remove('hidden');
            } else {
                yamlSaveStatus.textContent = `Error: ${data.message}`;
                yamlSaveStatus.className = 'status-msg error-msg';
                yamlSaveStatus.classList.remove('hidden');
            }
        } catch (e) {
            console.error(e);
            yamlSaveStatus.textContent = 'Error: Failed to save YAML config.';
            yamlSaveStatus.className = 'status-msg error-msg';
            yamlSaveStatus.classList.remove('hidden');
        } finally {
            btnYamlSave.textContent = 'Save YAML';
            btnYamlSave.disabled = false;
        }
    });

    loadDatasetYaml();
    dataYamlPath.addEventListener('change', loadDatasetYaml);

    // --- HELP CHATBOT DRAWER (❓ FAB) ---
    const floatingHelpBtn = document.getElementById('floating-help-btn');
    const helpDrawer = document.getElementById('help-drawer');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    const helpChatLog = document.getElementById('help-chat-log');
    const helpChatForm = document.getElementById('help-chat-form');
    const helpChatInput = document.getElementById('help-chat-input');
    const helpChatSend = document.getElementById('help-chat-send');
    let helpChatBusy = false;

    const isHelpOpen = () => helpDrawer && !helpDrawer.classList.contains('hidden');

    const setHelpOpen = (open) => {
        if (!helpDrawer || !floatingHelpBtn) return;
        helpDrawer.classList.toggle('hidden', !open);
        floatingHelpBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) {
            helpChatInput?.focus();
        }
    };

    const appendHelpMessage = (role, text, meta) => {
        if (!helpChatLog) return;
        const el = document.createElement('div');
        el.className = `help-chat-msg help-chat-${role === 'user' ? 'user' : 'bot'}`;
        el.style.cssText = role === 'user'
            ? 'padding: 10px 12px; border-radius: 8px; background: rgba(0,200,150,0.12); border: 1px solid var(--border); font-size: 13px; color: #e6e9ec; line-height: 1.5; white-space: pre-wrap;'
            : 'padding: 10px 12px; border-radius: 8px; background: #141a20; border: 1px solid var(--border); font-size: 13px; color: #e6e9ec; line-height: 1.5; white-space: pre-wrap;';
        el.textContent = text;
        if (meta) {
            const cap = document.createElement('div');
            cap.style.cssText = 'margin-top: 6px; font-size: 11px; color: var(--text-muted);';
            cap.textContent = meta;
            el.appendChild(cap);
        }
        helpChatLog.appendChild(el);
        helpChatLog.scrollTop = helpChatLog.scrollHeight;
    };

    const askHelpChat = async (raw) => {
        const question = (raw || '').trim();
        if (!question || helpChatBusy) return;
        helpChatBusy = true;
        if (helpChatSend) helpChatSend.disabled = true;
        appendHelpMessage('user', question);
        if (helpChatInput) helpChatInput.value = '';
        try {
            const res = await fetch(`${API_BASE}/api/help/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.detail || data.message || `Help API ${res.status}`);
            }
            const provider = data.provider === 'faq_offline' ? 'local FAQ index' : (data.provider || 'help');
            const sources = Array.isArray(data.sources) && data.sources.length
                ? `Sources: ${data.sources.map((s) => s.title).filter(Boolean).slice(0, 4).join(' · ')}`
                : null;
            appendHelpMessage('assistant', data.answer || 'No answer returned.', `via ${provider}${sources ? ` · ${sources}` : ''}`);
        } catch (err) {
            appendHelpMessage(
                'assistant',
                'Help API unreachable. Try again when the backend is up.\n\n'
                + 'Quick tips:\n'
                + '• Dataset Configuration → set YAML / source path → Save\n'
                + '• Ingest & Fusion → run ingest → browse pipeline frames\n'
                + '• Strict Execution Mode → Evidence cards / Execution Console (ledger under runs/executions/)',
                err instanceof Error ? err.message : String(err),
            );
        } finally {
            helpChatBusy = false;
            if (helpChatSend) helpChatSend.disabled = false;
        }
    };

    floatingHelpBtn?.addEventListener('click', () => setHelpOpen(!isHelpOpen()));
    closeDrawerBtn?.addEventListener('click', () => setHelpOpen(false));
    helpChatForm?.addEventListener('submit', (e) => {
        e.preventDefault();
        void askHelpChat(helpChatInput?.value || '');
    });
    document.querySelectorAll('.help-suggest-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            setHelpOpen(true);
            void askHelpChat(btn.getAttribute('data-q') || btn.textContent || '');
        });
    });

    // Populate initial default dataset if loaded
    if (selectDatasetType.value !== 'local') {
        loadNvidiaDataset(selectDatasetType.value);
    }

    // --- 3D PERCEPTION PIPELINE ---
    const btnIngestRun = document.getElementById('btn-ingest-run');
    const btnPerceptionRun = document.getElementById('btn-perception-run');
    const btnTrackingRun = document.getElementById('btn-tracking-run');
    const btnQualityGateRun = document.getElementById('btn-quality-gate-run');
    const btnLaunchGateRun = document.getElementById('btn-launch-gate-run');

    if (btnIngestRun) {
        btnIngestRun.addEventListener('click', async () => {
            const vendors = [];
            if (document.getElementById('ingest-vendor-local')?.checked) vendors.push('local');
            if (document.getElementById('ingest-vendor-alpamayo')?.checked) vendors.push('alpamayo');
            if (document.getElementById('ingest-vendor-waymo')?.checked) vendors.push('waymo');
            if (document.getElementById('ingest-vendor-a2d2')?.checked) vendors.push('a2d2');
            if (!vendors.length) {
                document.getElementById('ingest-status').textContent =
                    'Select at least one vendor (Local for real sequences, or Alpamayo/Waymo/A2D2 demo stubs).';
                return;
            }
            currentSequenceId = document.getElementById('ingest-sequence-id')?.value || 'seq_001';
            const sourcePath = document.getElementById('ingest-source-path')?.value
                || document.getElementById('dataset-source-path')?.value || document.getElementById('input-source')?.value
                || 'data';
            const maxFramesRaw = document.getElementById('ingest-max-frames')?.value;
            const maxFrames = maxFramesRaw ? Number(maxFramesRaw) : undefined;
            const statusEl = document.getElementById('ingest-status');
            statusEl.textContent = 'Running ingest & fusion...';
            try {
                const res = await fetch(`${API_BASE}/api/dataset/ingest`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        vendors,
                        sequence_id: currentSequenceId,
                        source_path: sourcePath,
                        max_frames: Number.isFinite(maxFrames) ? maxFrames : undefined,
                    }),
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    const stubTag = data.demo_stub ? 'demo stub: ' : '';
                    statusEl.textContent =
                        `Ingested ${stubTag}${data.frames} frames (${data.vendor || 'mixed'}) -> ${data.manifest}`;
                } else {
                    statusEl.textContent = `Error: ${data.detail || JSON.stringify(data)}`;
                }
                await refreshPipelineStatus();
                await refreshAllOutputBrowsers();
            } catch (e) {
                statusEl.textContent = `Error: ${e.message}`;
            }
        });
    }

    if (btnPerceptionRun) {
        btnPerceptionRun.addEventListener('click', async () => {
            const statusEl = document.getElementById('perception-status');
            statusEl.textContent = 'Running SAM auto-label...';
            try {
                const res = await fetch(`${API_BASE}/api/perception/auto-label`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sequence_id: currentSequenceId,
                        sam_checkpoint: document.getElementById('perception-sam-checkpoint')?.value || 'models/sam_vit_b.pth',
                        device: document.getElementById('perception-device')?.value || 'cpu',
                        no_sam: !!document.getElementById('perception-no-sam')?.checked,
                    }),
                });
                const data = await res.json();
                if (!res.ok) {
                    statusEl.textContent = `Error: ${data.detail?.message || data.detail || JSON.stringify(data)}`;
                    renderEvidenceCard('evidence-perception', {
                        status: 'FAILED',
                        execution_id: data.detail?.execution_id || data.execution_id,
                        message: data.detail?.message || data.detail,
                    });
                } else {
                    const stubPrefix = data.demo_stub ? 'demo stub: ' : '';
                    const expected = data.frames_expected != null ? `/${data.frames_expected}` : '';
                    statusEl.textContent =
                        `${data.status}: ${stubPrefix}Processed ${data.frames_processed}${expected} frames → ${data.proposals_dir}`;
                    renderEvidenceCard('evidence-perception', {
                        status: data.status,
                        execution_id: data.execution_id,
                        duration_ms: data.duration_ms,
                        model: data.model,
                        checkpoint: data.checkpoint,
                        metrics: {
                            frames_processed: data.frames_processed,
                            frames_expected: data.frames_expected,
                            predictions_generated: data.predictions_generated,
                            inference_calls: data.inference_calls,
                            sam_ran: data.sam_ran,
                            output_dir: data.proposals_dir,
                        },
                        message: data.message,
                        events: data.events,
                    });
                    refreshExecutionConsole();
                }
                await refreshPipelineStatus();
                await refreshAllOutputBrowsers();
            } catch (e) {
                statusEl.textContent = `Error: ${e.message}`;
            }
        });
    }

    // --- PIPELINE OUTPUT BROWSER ---
    function initOutputBrowser(root) {
        const seqSelect = root.querySelector('.out-seq-select');
        const frameList = root.querySelector('.out-frame-list');
        const summary = root.querySelector('.out-summary');
        const previewImg = root.querySelector('.out-preview-img');
        const previewMeta = root.querySelector('.out-preview-meta');
        const proposalsPre = root.querySelector('.out-proposals-json');
        const refreshBtn = root.querySelector('.out-refresh-btn');

        async function loadSequences(preferredId) {
            try {
                const res = await fetch(`${API_BASE}/api/pipeline/sequences`);
                const data = await res.json();
                const sequences = data.sequences || [];
                const previous = preferredId || seqSelect.value || currentSequenceId;
                seqSelect.innerHTML = '';
                if (!sequences.length) {
                    const opt = document.createElement('option');
                    opt.value = currentSequenceId;
                    opt.textContent = `${currentSequenceId} (empty)`;
                    seqSelect.appendChild(opt);
                } else {
                    sequences.forEach((s) => {
                        const opt = document.createElement('option');
                        opt.value = s.sequence_id;
                        const stub = s.demo_stub ? ' stub' : '';
                        opt.textContent = `${s.sequence_id} · ${s.frames ?? 0} frames${stub}`;
                        seqSelect.appendChild(opt);
                    });
                }
                if ([...seqSelect.options].some((o) => o.value === previous)) {
                    seqSelect.value = previous;
                }
            } catch (e) {
                if (summary) summary.textContent = `Could not list sequences: ${e.message}`;
            }
        }

        async function loadFrames() {
            const seqId = seqSelect.value || currentSequenceId;
            currentSequenceId = seqId;
            if (summary) summary.textContent = `Loading ${seqId}…`;
            frameList.innerHTML = '';
            previewImg.style.display = 'none';
            proposalsPre.style.display = 'none';
            previewMeta.textContent = 'No frame selected.';
            try {
                const res = await fetch(`${API_BASE}/api/pipeline/frames?sequence_id=${encodeURIComponent(seqId)}`);
                const data = await res.json();
                if (!data.browsable) {
                    summary.textContent = data.empty_reason || 'Nothing browsable for this sequence.';
                    return;
                }
                const stub = data.demo_stub ? ' · demo stub' : '';
                const tracks = data.has_tracks ? ' · tracks.json' : '';
                summary.textContent =
                    `${data.frames.length} frame(s) · vendor ${data.vendor || '—'}${stub} · ` +
                    `${data.proposal_files || 0} proposal file(s)${tracks} · ${data.base_path}`;
                data.frames.forEach((frame) => {
                    const btn = document.createElement('button');
                    btn.type = 'button';
                    btn.className = 'out-frame-item';
                    btn.innerHTML = `<div>${frame.frame_id}</div><div class="muted">${frame.proposal_count} proposals · ${frame.cameras?.length || 0} cam</div>`;
                    btn.addEventListener('click', () => {
                        frameList.querySelectorAll('.out-frame-item').forEach((el) => el.classList.remove('active'));
                        btn.classList.add('active');
                        void showFrame(seqId, frame);
                    });
                    frameList.appendChild(btn);
                });
            } catch (e) {
                summary.textContent = `Failed to load frames: ${e.message}`;
            }
        }

        async function showFrame(seqId, frameSummary) {
            previewMeta.textContent = `Loading ${frameSummary.frame_id}…`;
            try {
                const res = await fetch(
                    `${API_BASE}/api/pipeline/frame?sequence_id=${encodeURIComponent(seqId)}&frame_id=${encodeURIComponent(frameSummary.frame_id)}`
                );
                const data = await res.json();
                const url = (data.cameras || []).find((c) => c.preview_url)?.preview_url || frameSummary.preview_url;
                if (url) {
                    previewImg.src = url;
                    previewImg.style.display = 'block';
                } else {
                    previewImg.style.display = 'none';
                }
                previewMeta.textContent =
                    `${data.frame_id} · ${data.proposal_count} proposals` +
                    (data.proposals_path ? ` · ${data.proposals_path}` : ' · no proposals yet');
                if (data.proposals && (Array.isArray(data.proposals) ? data.proposals.length : true)) {
                    proposalsPre.style.display = 'block';
                    proposalsPre.textContent = JSON.stringify(data.proposals, null, 2);
                } else {
                    proposalsPre.style.display = 'block';
                    proposalsPre.textContent = '// No proposals for this frame yet. Run Auto-Label.';
                }
            } catch (e) {
                previewMeta.textContent = `Failed to load frame: ${e.message}`;
            }
        }

        seqSelect.addEventListener('change', () => { void loadFrames(); });
        refreshBtn.addEventListener('click', async () => {
            await loadSequences(currentSequenceId);
            await loadFrames();
        });

        root._refreshOutputs = async () => {
            await loadSequences(currentSequenceId);
            await loadFrames();
        };

        void root._refreshOutputs();
    }

    const outputBrowsers = [...document.querySelectorAll('.pipeline-output-browser')];
    outputBrowsers.forEach(initOutputBrowser);

    async function refreshAllOutputBrowsers() {
        for (const root of outputBrowsers) {
            if (typeof root._refreshOutputs === 'function') {
                await root._refreshOutputs();
            }
        }
    }

    if (btnTrackingRun) {
        btnTrackingRun.addEventListener('click', async () => {
            const statusEl = document.getElementById('tracking-status');
            statusEl.textContent = 'Running temporal tracker...';
            try {
                const res = await fetch(`${API_BASE}/api/perception/track`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence_id: currentSequenceId }),
                });
                const data = await res.json();
                statusEl.textContent = data.status === 'ok'
                    ? `Tracked ${data.num_tracks} objects -> ${data.tracks_file}`
                    : `Error: ${data.detail || JSON.stringify(data)}`;
                await refreshPipelineStatus();
            } catch (e) {
                statusEl.textContent = `Error: ${e.message}`;
            }
        });
    }

    if (btnQualityGateRun) {
        btnQualityGateRun.addEventListener('click', async () => {
            const statusEl = document.getElementById('quality-gate-status');
            const metricsEl = document.getElementById('quality-gate-metrics');
            statusEl.textContent = 'Running quality gate benchmark...';
            try {
                const res = await fetch(`${API_BASE}/api/gates/quality`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence_id: currentSequenceId }),
                });
                const data = await res.json();
                if (data.metric_card) {
                    statusEl.textContent = data.passed ? 'Quality gate PASSED' : 'Quality gate FAILED';
                    metricsEl.innerHTML = `<pre style="background:rgba(0,0,0,0.3);padding:12px;border-radius:8px;overflow:auto;">${JSON.stringify(data.metric_card, null, 2)}</pre>`;
                } else {
                    statusEl.textContent = `Error: ${data.detail || JSON.stringify(data)}`;
                }
                await refreshPipelineStatus();
            } catch (e) {
                statusEl.textContent = `Error: ${e.message}`;
            }
        });
    }

    if (btnLaunchGateRun) {
        btnLaunchGateRun.addEventListener('click', async () => {
            const statusEl = document.getElementById('launch-gate-status');
            const badgeEl = document.getElementById('launch-gate-badge');
            statusEl.textContent = 'Evaluating launch gate...';
            try {
                const res = await fetch(`${API_BASE}/api/gates/launch`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ sequence_id: currentSequenceId }),
                });
                const data = await res.json();
                statusEl.textContent = data.passed ? 'Launch gate PASSED - export allowed' : 'Launch gate FAILED - export blocked';
                badgeEl.innerHTML = data.passed
                    ? '<span style="color:#00ffaa;font-weight:600;">EXPORT AUTHORIZED</span>'
                    : `<span style="color:#ff4444;font-weight:600;">EXPORT BLOCKED</span><pre style="margin-top:8px;">${JSON.stringify(data.failures, null, 2)}</pre>`;
                await refreshPipelineStatus();
            } catch (e) {
                statusEl.textContent = `Error: ${e.message}`;
            }
        });
    }

    // --- Execution integrity: evidence cards, console, strict mode, load ---
    function esc(s) {
        return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function renderEvidenceCard(elementId, evidence) {
        const el = document.getElementById(elementId);
        if (!el || !evidence) return;
        const status = evidence.status || 'NOT_EXECUTED';
        const metrics = evidence.metrics || {};
        const metricRows = Object.entries(metrics).map(([k, v]) => {
            const val = formatMetricValue(k, v);
            return `<div class="ev-row"><span>${esc(k)}</span><strong>${esc(val)}</strong></div>`;
        }).join('');
        const events = (evidence.events || []).slice(-8).map(ev =>
            `<li><code>${esc(ev.ts || '')}</code> ${esc(ev.kind || '')}: ${esc(ev.message || '')}</li>`
        ).join('');
        const ckpt = evidence.checkpoint
            ? `<div class="ev-row"><span>checkpoint</span><strong>${esc(evidence.checkpoint.path || evidence.checkpoint)} ${evidence.checkpoint.exists === false ? '(missing)' : ''}</strong></div>`
            : '';
        el.innerHTML = `
            <h4>What actually happened?</h4>
            <div class="ev-status ev-status-${esc(status)}">${esc(status)}</div>
            <div class="ev-row"><span>execution_id</span><strong class="font-mono">${esc(evidence.execution_id || '—')}</strong></div>
            ${evidence.duration_ms != null ? `<div class="ev-row"><span>duration_ms</span><strong>${esc(evidence.duration_ms)}</strong></div>` : ''}
            ${evidence.model ? `<div class="ev-row"><span>model</span><strong>${esc(evidence.model)}</strong></div>` : ''}
            ${ckpt}
            ${evidence.process_id != null ? `<div class="ev-row"><span>process_id</span><strong>${esc(evidence.process_id)}</strong></div>` : ''}
            ${evidence.exit_code != null ? `<div class="ev-row"><span>exit_code</span><strong>${esc(evidence.exit_code)}</strong></div>` : ''}
            ${evidence.command ? `<div class="ev-row"><span>command</span><strong class="font-mono">${esc(Array.isArray(evidence.command) ? evidence.command.join(' ') : evidence.command)}</strong></div>` : ''}
            ${metricRows}
            ${evidence.message ? `<p class="ev-msg">${esc(evidence.message)}</p>` : ''}
            ${evidence.execution_id ? `<a class="ev-log-link" href="/api/executions/${encodeURIComponent(evidence.execution_id)}/log" target="_blank" rel="noopener">Open execution log</a>` : ''}
            ${events ? `<details class="ev-timeline"><summary>Event timeline</summary><ul>${events}</ul></details>` : ''}
        `;
    }

    function refreshScriptStatus(scripts) {
        const map = {
            'train.py': 'script-status-train',
            'infer.py': 'script-status-infer',
            'autograder.py': 'script-status-grader',
        };
        Object.entries(map).forEach(([name, id]) => {
            const el = document.getElementById(id);
            const r = scripts?.[name];
            if (!el || !r) return;
            if (!r.exists) {
                el.textContent = `Missing (${name})`;
                el.className = 'val font-mono';
                el.style.color = '#ff8888';
            } else if (r.syntax_valid && r.dry_run_ok) {
                el.textContent = `Verified (${name})${r.last_status ? ' · last ' + r.last_status : ''}`;
                el.className = 'val font-mono green';
                el.style.color = '';
            } else {
                el.textContent = `Found but unverified (${name})`;
                el.className = 'val font-mono';
                el.style.color = '#fbbf24';
            }
        });
    }

    async function refreshHealth() {
        try {
            const res = await fetch(`${API_BASE}/api/health`);
            const data = await res.json();
            const chip = document.getElementById('health-status-chip');
            if (chip) {
                chip.textContent = [
                    data.filesystem_writable ? 'FS writable' : 'FS FAIL',
                    data.torch_installed ? 'torch' : 'no-torch',
                    data.ultralytics_installed ? 'yolo' : 'no-yolo',
                    data.gpu?.available ? `GPU:${data.gpu.name}` : 'CPU',
                    data.last_successful_execution_id ? `last:${data.last_successful_execution_id}` : 'no-success-yet',
                ].join(' · ');
            }
            if (data.scripts) refreshScriptStatus(data.scripts);
            const tog = document.getElementById('toggle-strict-mode');
            if (tog) tog.checked = !!data.strict_mode;
        } catch (e) {
            const chip = document.getElementById('health-status-chip');
            if (chip) chip.textContent = 'Backend unreachable';
        }
    }

    async function refreshExecutionConsole() {
        const list = document.getElementById('exec-console-list');
        if (!list) return;
        try {
            const res = await fetch(`${API_BASE}/api/executions?limit=40`);
            const data = await res.json();
            const rows = data.executions || [];
            if (!rows.length) {
                list.innerHTML = '<p class="evidence-empty">No executions recorded yet.</p>';
                return;
            }
            list.innerHTML = rows.map(e => `
                <button type="button" class="exec-row" data-id="${esc(e.execution_id)}">
                    <span class="ev-status ev-status-${esc(e.status)}">${esc(e.status)}</span>
                    <span class="font-mono">${esc(e.execution_id)}</span>
                    <span>${esc(e.operation)}</span>
                    <span>${esc(e.duration_ms != null ? e.duration_ms + 'ms' : '—')}</span>
                    <span>${e.verified ? 'verified' : 'unverified'}</span>
                </button>
            `).join('');
            list.querySelectorAll('.exec-row').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const detail = document.getElementById('exec-console-detail');
                    const r = await fetch(`${API_BASE}/api/executions/${encodeURIComponent(btn.dataset.id)}`);
                    const full = await r.json();
                    detail.classList.remove('hidden');
                    detail.innerHTML = `<pre>${esc(JSON.stringify(full, null, 2))}</pre>`;
                });
            });
        } catch (e) {
            list.innerHTML = `<p class="evidence-empty">Failed to load executions: ${esc(e.message)}</p>`;
        }
    }

    const btnLoadPreprocess = document.getElementById('btn-load-preprocess');
    if (btnLoadPreprocess) {
        btnLoadPreprocess.addEventListener('click', async () => {
            btnLoadPreprocess.disabled = true;
            btnLoadPreprocess.textContent = 'Loading…';
            const statusEl = document.getElementById('source-browse-status');
            try {
                const res = await fetch(`${API_BASE}/api/dataset/load`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        source_path: datasetSourcePath?.value || 'data',
                        yaml_path: dataYamlPath?.value || 'coco8.yaml',
                        dataset_type: selectDatasetType?.value || 'local',
                    }),
                });
                const data = await res.json();
                const m = data.metrics || {};
                if (statusEl) {
                    statusEl.textContent = `${data.status}: discovered=${m.images_discovered ?? 0} readable=${m.images_readable ?? 0} corrupt=${m.images_corrupt ?? 0} → ${data.manifest_path || ''}`;
                }
                renderEvidenceCard('evidence-dataset', {
                    status: data.status,
                    execution_id: data.execution_id,
                    duration_ms: data.duration_ms,
                    metrics: {
                        ...m,
                        reconciliation: data.reconciliation?.state,
                        manifest: data.manifest_path,
                    },
                    message: data.message,
                    events: data.events,
                });
                const badge = document.getElementById('ds-meta-badge');
                const loadedEl = document.getElementById('ds-kpi-loaded-rows');
                const pctEl = document.getElementById('ds-kpi-ingest-pct');
                if (loadedEl) loadedEl.textContent = m.images_readable ?? 0;
                if (pctEl) {
                    pctEl.textContent = m.loaded_pct_of_discovered != null
                        ? `${formatPercent(m.loaded_pct_of_discovered, { alreadyPercent: true })} of discovered`
                        : '—';
                }
                if (badge) {
                    if (data.status === 'FAILED' || (m.images_readable ?? 0) === 0) {
                        badge.textContent = 'FAILED — 0 loaded';
                        badge.style.background = 'rgba(255, 68, 68, 0.12)';
                        badge.style.color = '#ff8888';
                    } else {
                        const pctLabel = m.loaded_pct_of_discovered != null
                            ? formatPercent(m.loaded_pct_of_discovered, { alreadyPercent: true })
                            : null;
                        badge.textContent = pctLabel
                            ? `${pctLabel} Loaded`
                            : `${formatNumber(m.images_readable)} loaded (evidence)`;
                        badge.style.background = 'rgba(0, 255, 170, 0.15)';
                        badge.style.color = '#00ffaa';
                    }
                }
                const cfgStatus = document.getElementById('config-exec-status');
                if (cfgStatus) {
                    cfgStatus.textContent = `used_by_execution YES · last execution_id ${data.execution_id}`;
                    cfgStatus.dataset.modified = '0';
                }
                refreshExecutionConsole();
                if (data.status === 'FAILED' || data.status === 'VALIDATION_FAILED') {
                    notify(data.message || 'Load failed', 'error');
                } else if (data.status === 'PARTIAL_SUCCESS') {
                    notify(data.message || 'Partial load', 'warning');
                } else {
                    notify(data.message || 'Load complete', 'success');
                }
            } catch (e) {
                if (statusEl) statusEl.textContent = `Load failed: ${e.message}`;
                notify('Load & Preprocess failed.', 'error');
            } finally {
                btnLoadPreprocess.disabled = false;
                btnLoadPreprocess.textContent = 'Load & Preprocess';
            }
        });
    }

    const btnYamlValidate = document.getElementById('btn-yaml-validate');
    if (btnYamlValidate) {
        btnYamlValidate.addEventListener('click', async () => {
            try {
                const res = await fetch(`${API_BASE}/api/yaml/validate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: dataYamlPath?.value || 'coco8.yaml' }),
                });
                const data = await res.json();
                notify(`${data.status}: classes=${data.class_count} images=${JSON.stringify(data.image_counts)}`, data.status === 'SUCCEEDED' ? 'success' : 'warning');
                const cfgStatus = document.getElementById('config-exec-status');
                if (cfgStatus) {
                    cfgStatus.textContent = `YAML hash ${data.content_hash || '—'} · ${data.status} · execution ${data.execution_id || '—'}`;
                }
                refreshExecutionConsole();
            } catch (e) {
                notify(`YAML validate failed: ${e.message}`, 'error');
            }
        });
    }

    document.getElementById('btn-open-exec-console')?.addEventListener('click', () => {
        document.getElementById('exec-console-modal')?.classList.remove('hidden');
        refreshExecutionConsole();
    });
    document.getElementById('btn-close-exec-console')?.addEventListener('click', () => {
        document.getElementById('exec-console-modal')?.classList.add('hidden');
    });
    document.getElementById('btn-refresh-executions')?.addEventListener('click', refreshExecutionConsole);
    document.getElementById('btn-health-check')?.addEventListener('click', async () => {
        await refreshHealth();
        notify('Health refreshed — see Workspace Status chip.', 'info');
    });
    document.getElementById('toggle-strict-mode')?.addEventListener('change', async (e) => {
        try {
            await fetch(`${API_BASE}/api/strict-mode`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: e.target.checked }),
            });
            notify(e.target.checked ? 'Strict Execution Mode ON' : 'Strict Execution Mode OFF', 'info');
        } catch (err) {
            notify('Failed to toggle strict mode', 'error');
        }
    });

    refreshHealth();
});
