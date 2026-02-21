#!/usr/bin/env python3
"""
Training dashboard for YOLOv8-seg (FinnWoodlands). Run on localhost to track
epoch, elapsed time, ETA, and metrics. Polls results.csv written by Ultralytics.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timedelta

from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
RESULTS_DIR = _REPO_ROOT / "segmentation_results" / "bark_segmentation"
TOTAL_EPOCHS = 100  # set via --epochs; used for ETA
UPDATE_INTERVAL_SEC = 2


def find_results_csv(results_dir: Path):
    """Find results.csv; may be in results_dir or results_dir/weights/.. (Ultralytics saves in run folder)."""
    if not results_dir.exists():
        return None
    for name in ("results.csv", "results.csv.bak"):
        c = results_dir / name
        if c.exists():
            return c
    for sub in sorted(results_dir.iterdir()):
        if sub.is_dir() and (sub / "results.csv").exists():
            return sub / "results.csv"
    return None


def load_progress(results_dir: Path, total_epochs: int):
    """Load current epoch, cumulative time, and compute ETA. Returns dict or None."""
    csv_path = find_results_csv(results_dir)
    if not csv_path:
        return None
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return None
    if not rows:
        return {"epoch": 0, "total_epochs": total_epochs, "elapsed_sec": 0, "eta_sec": None, "metrics": {}}
    last = rows[-1]
    try:
        epoch = int(float(last.get("epoch", len(rows))))
    except (ValueError, TypeError):
        epoch = len(rows)
    try:
        elapsed_sec = float(last.get("time", 0))
    except (ValueError, TypeError):
        elapsed_sec = 0
    # ETA: average time per epoch * remaining
    remaining = max(0, total_epochs - epoch)
    if epoch > 0 and elapsed_sec > 0 and remaining > 0:
        sec_per_epoch = elapsed_sec / epoch
        eta_sec = sec_per_epoch * remaining
    else:
        eta_sec = None
    # Metrics (segmentation columns may vary)
    metrics = {}
    for key in ("train/box_loss", "train/seg_loss", "train/cls_loss", "val/box_loss", "val/seg_loss", "val/cls_loss",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)", "train/loss", "val/loss"):
        if key in last:
            try:
                metrics[key] = float(last[key])
            except (ValueError, TypeError):
                pass
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "remaining_epochs": remaining,
        "metrics": metrics,
        "csv_path": str(csv_path),
    }


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8-seg Training Dashboard</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 900px; margin: 0 auto; padding: 24px; }
        h1 { font-size: 1.5rem; margin-bottom: 8px; color: #a8e6cf; }
        .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 24px; }
        .card { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .progress-section { margin-bottom: 24px; }
        .progress-bar-wrap { height: 28px; background: #0f3460; border-radius: 14px; overflow: hidden; margin: 8px 0; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #a8e6cf, #56ab91); border-radius: 14px; transition: width 0.5s; }
        .progress-labels { display: flex; justify-content: space-between; font-size: 0.85rem; color: #aaa; margin-top: 4px; }
        .stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
        .stat { background: #0f3460; padding: 12px; border-radius: 8px; }
        .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; margin-bottom: 4px; }
        .stat-value { font-size: 1.25rem; font-weight: 600; color: #a8e6cf; }
        .stat-value.warn { color: #f9c74f; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }
        .metric { font-size: 0.8rem; color: #bbb; }
        .metric span { color: #fff; font-weight: 500; }
        .status { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #f9c74f; animation: pulse 2s infinite; }
        .status.done { background: #a8e6cf; animation: none; }
        .status.error { background: #e63946; animation: none; }
        @keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:0.5 } }
        .footer { font-size: 0.75rem; color: #666; margin-top: 24px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv8-seg Training</h1>
        <p class="subtitle">FinnWoodlands Spruce + Pine · Dashboard updates every 2s</p>

        <div class="card progress-section">
            <div class="progress-labels">
                <span>Epoch <strong id="epochCur">0</strong> / <strong id="epochTot">100</strong></span>
                <span id="etaText">ETA: --</span>
            </div>
            <div class="progress-bar-wrap">
                <div class="progress-bar" id="progressBar" style="width: 0%;"></div>
            </div>
            <div class="progress-labels">
                <span id="elapsedText">Elapsed: 0:00</span>
                <span id="remainingText">Remaining: --</span>
            </div>
        </div>

        <div class="card">
            <div class="stat-label"><span class="status" id="statusDot"></span> Status</div>
            <div class="stat-value" id="statusText">Waiting for results...</div>
            <div class="metrics" id="metricsBox"></div>
        </div>

        <div class="footer">
            Results path: <code id="csvPath">--</code>
        </div>
    </div>
    <script>
        function fmtTime(sec) {
            if (sec == null || isNaN(sec)) return '--';
            var m = Math.floor(sec / 60), s = Math.floor(sec % 60);
            return m + ':' + (s < 10 ? '0' : '') + s;
        }
        function update(data) {
            var epoch = data.epoch || 0;
            var total = data.total_epochs || 100;
            var pct = total > 0 ? Math.min(100, (epoch / total) * 100) : 0;
            document.getElementById('epochCur').textContent = epoch;
            document.getElementById('epochTot').textContent = total;
            document.getElementById('progressBar').style.width = pct + '%';
            document.getElementById('elapsedText').textContent = 'Elapsed: ' + fmtTime(data.elapsed_sec);
            document.getElementById('remainingText').textContent = 'Remaining: ' + fmtTime(data.eta_sec);
            document.getElementById('etaText').textContent = 'ETA: ' + fmtTime(data.eta_sec);
            document.getElementById('csvPath').textContent = data.csv_path || '--';
            var dot = document.getElementById('statusDot');
            var status = document.getElementById('statusText');
            if (epoch >= total && total > 0) {
                dot.className = 'status done';
                status.textContent = 'Training complete';
            } else if (data.epoch > 0) {
                dot.className = 'status';
                status.textContent = 'Training in progress';
            } else {
                dot.className = 'status';
                status.textContent = 'Waiting for results...';
            }
            var m = data.metrics || {};
            var html = '';
            for (var k in m) {
                var v = m[k];
                var label = k.replace(/^.*\//, '').replace(/\(B\)$/, '');
                html += '<div class="metric">' + label + ': <span>' + (typeof v === 'number' ? v.toFixed(4) : v) + '</span></div>';
            }
            document.getElementById('metricsBox').innerHTML = html || '<div class="metric">No metrics yet</div>';
        }
        function fetchData() {
            fetch('/api/progress').then(function(r) { return r.json(); }).then(update).catch(function() {
                update({ epoch: 0, total_epochs: {{ total_epochs }}, elapsed_sec: 0, eta_sec: null, metrics: {}, csv_path: null });
            });
        }
        fetchData();
        setInterval(fetchData, 2000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, total_epochs=app.config.get("TOTAL_EPOCHS", 100))


@app.route("/api/progress")
def api_progress():
    results_dir = app.config.get("RESULTS_DIR", RESULTS_DIR)
    total_epochs = app.config.get("TOTAL_EPOCHS", TOTAL_EPOCHS)
    data = load_progress(results_dir, total_epochs)
    if data is None:
        return jsonify({
            "epoch": 0,
            "total_epochs": total_epochs,
            "elapsed_sec": 0,
            "eta_sec": None,
            "remaining_epochs": total_epochs,
            "metrics": {},
            "csv_path": None,
        })
    return jsonify(data)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-seg training dashboard (localhost)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory containing results.csv (e.g. project/name or data/runs/exp)")
    parser.add_argument("--project", type=Path, default=None,
                        help="Project directory (e.g. segmentation_results); name will be first run dir")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs (for ETA)")
    parser.add_argument("--port", type=int, default=5010, help="Port (default 5010)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    args = parser.parse_args()
    if args.results_dir is not None:
        results_dir = args.results_dir.resolve()
    elif args.project is not None:
        results_dir = args.project.resolve()
    else:
        results_dir = (Path(__file__).resolve().parent.parent / "data" / "runs").resolve()
    app.config["RESULTS_DIR"] = results_dir
    app.config["TOTAL_EPOCHS"] = args.epochs
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Results dir: {results_dir} (create this when you start training)")
    print(f"ETA based on total epochs: {args.epochs}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
