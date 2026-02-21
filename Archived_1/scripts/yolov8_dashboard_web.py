# yolov8_dashboard_web.py
# Web-based dashboard for YOLOv8 training metrics
# Displays: train loss, val loss, mAP, precision, recall, precision-recall curve

from flask import Flask, render_template_string, jsonify
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

app = Flask(__name__)

RESULTS_DIR = Path('../data/models/yolov8_results/bark_classifier')
UPDATE_INTERVAL = 2

def find_results_csv():
    """Find the results.csv file from YOLOv8 training"""
    if not RESULTS_DIR.exists():
        return None
    
    csv_file = RESULTS_DIR / "results.csv"
    if csv_file.exists():
        return csv_file
    return None

def load_yolov8_metrics():
    """Load metrics from YOLOv8 results.csv file"""
    csv_file = find_results_csv()
    
    if not csv_file:
        return None
    
    try:
        df = pd.read_csv(csv_file)
        
        # Extract metrics
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map': []
        }
        
        for idx, row in df.iterrows():
            epoch = int(row.get('epoch', idx + 1))
            metrics['epochs'].append(epoch)
            
            # Training loss
            train_loss = row.get('train/loss', row.get('train_loss', None))
            if train_loss is not None and not pd.isna(train_loss):
                metrics['train_loss'].append(float(train_loss))
            else:
                metrics['train_loss'].append(None)
            
            # Validation loss
            val_loss = row.get('val/loss', None)
            if val_loss is not None and not pd.isna(val_loss):
                metrics['val_loss'].append(float(val_loss))
            else:
                # Use train loss if val loss not available (classification mode)
                metrics['val_loss'].append(metrics['train_loss'][-1] if metrics['train_loss'] else None)
            
            # Accuracy (top1)
            train_acc = row.get('metrics/accuracy_top1', row.get('train/accuracy_top1', None))
            val_acc = row.get('metrics/accuracy_top1', row.get('val/accuracy_top1', None))
            
            if train_acc is not None and not pd.isna(train_acc):
                metrics['train_acc'].append(float(train_acc) * 100)
            else:
                metrics['train_acc'].append(None)
            
            if val_acc is not None and not pd.isna(val_acc):
                metrics['val_acc'].append(float(val_acc) * 100)
            else:
                metrics['val_acc'].append(None)
            
            # Precision and Recall
            precision = row.get('metrics/precision(B)', row.get('metrics/precision', row.get('precision', None)))
            recall = row.get('metrics/recall(B)', row.get('metrics/recall', row.get('recall', None)))
            map50 = row.get('metrics/mAP50(B)', row.get('metrics/mAP50', row.get('mAP50', None)))
            map_val = row.get('metrics/mAP50-95(B)', row.get('metrics/mAP50-95', row.get('mAP', None)))
            
            metrics['precision'].append(float(precision) if precision is not None and not pd.isna(precision) else None)
            metrics['recall'].append(float(recall) if recall is not None and not pd.isna(recall) else None)
            metrics['map50'].append(float(map50) if map50 is not None and not pd.isna(map50) else None)
            metrics['map'].append(float(map_val) if map_val is not None and not pd.isna(map_val) else None)
        
        # Filter out None values for cleaner data
        for key in metrics:
            if key != 'epochs':
                metrics[key] = [x for x in metrics[key] if x is not None]
        
        return metrics
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv8 Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .status-bar {
            display: flex;
            justify-content: space-around;
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .status-item {
            text-align: center;
        }
        .status-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .status-value {
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            position: relative;
            height: 400px;
            background-color: #fafafa;
            padding: 10px;
            border-radius: 5px;
        }
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌲 YOLOv8 Bark Classification Training Dashboard</h1>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-label">Status</div>
                <div class="status-value" id="status">Loading...</div>
            </div>
            <div class="status-item">
                <div class="status-label">Current Epoch</div>
                <div class="status-value" id="epoch">0</div>
            </div>
            <div class="status-item">
                <div class="status-label">Last Update</div>
                <div class="status-value" id="lastUpdate">Never</div>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Train Loss</div>
                <div class="metric-value" id="trainLoss">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Val Loss</div>
                <div class="metric-value" id="valLoss">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">mAP@0.5</div>
                <div class="metric-value" id="map50">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">mAP@0.5:0.95</div>
                <div class="metric-value" id="map">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value" id="precision">--</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value" id="recall">--</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <canvas id="lossChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="mapChart"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="precisionRecallChart"></canvas>
            </div>
        </div>
        
        <div class="controls">
            <button onclick="updateDashboard()">Refresh Now</button>
            <label style="margin-left: 20px;">
                <input type="checkbox" id="autoRefresh" checked onchange="toggleAutoRefresh()">
                Auto-refresh (every 2 seconds)
            </label>
        </div>
    </div>
    
    <script>
        let lossChart, mapChart, precisionRecallChart;
        let autoRefreshInterval;
        
        // Initialize Loss Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Train Loss',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }, {
                    label: 'Val Loss',
                    data: [],
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Training and Validation Loss'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        
        // Initialize mAP Chart
        const mapCtx = document.getElementById('mapChart').getContext('2d');
        mapChart = new Chart(mapCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'mAP@0.5',
                    data: [],
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                    tension: 0.1
                }, {
                    label: 'mAP@0.5:0.95',
                    data: [],
                    borderColor: 'rgb(255, 206, 86)',
                    backgroundColor: 'rgba(255, 206, 86, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Mean Average Precision (mAP)'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
        
        // Initialize Precision-Recall Chart
        const prCtx = document.getElementById('precisionRecallChart').getContext('2d');
        precisionRecallChart = new Chart(prCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Precision',
                    data: [],
                    borderColor: 'rgb(153, 102, 255)',
                    backgroundColor: 'rgba(153, 102, 255, 0.2)',
                    tension: 0.1,
                    yAxisID: 'y'
                }, {
                    label: 'Recall',
                    data: [],
                    borderColor: 'rgb(255, 159, 64)',
                    backgroundColor: 'rgba(255, 159, 64, 0.2)',
                    tension: 0.1,
                    yAxisID: 'y'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Precision vs Recall'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0
                    }
                }
            }
        });
        
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('status').textContent = data.error;
                        return;
                    }
                    
                    document.getElementById('status').textContent = 'Monitoring training...';
                    document.getElementById('epoch').textContent = data.epoch;
                    document.getElementById('lastUpdate').textContent = data.last_update;
                    
                    // Update metrics
                    if (data.train_loss !== null) {
                        document.getElementById('trainLoss').textContent = data.train_loss.toFixed(4);
                    }
                    if (data.val_loss !== null) {
                        document.getElementById('valLoss').textContent = data.val_loss.toFixed(4);
                    }
                    if (data.map50 !== null) {
                        document.getElementById('map50').textContent = data.map50.toFixed(4);
                    }
                    if (data.map !== null) {
                        document.getElementById('map').textContent = data.map.toFixed(4);
                    }
                    if (data.precision !== null) {
                        document.getElementById('precision').textContent = data.precision.toFixed(4);
                    }
                    if (data.recall !== null) {
                        document.getElementById('recall').textContent = data.recall.toFixed(4);
                    }
                    
                    // Update Loss Chart
                    if (data.epochs && data.epochs.length > 0) {
                        lossChart.data.labels = data.epochs;
                        lossChart.data.datasets[0].data = data.train_losses || [];
                        lossChart.data.datasets[1].data = data.val_losses || [];
                        lossChart.update();
                        
                        // Update mAP Chart
                        mapChart.data.labels = data.epochs;
                        mapChart.data.datasets[0].data = data.map50_values || [];
                        mapChart.data.datasets[1].data = data.map_values || [];
                        mapChart.update();
                        
                        // Update Precision-Recall Chart
                        precisionRecallChart.data.labels = data.epochs;
                        precisionRecallChart.data.datasets[0].data = data.precision_values || [];
                        precisionRecallChart.data.datasets[1].data = data.recall_values || [];
                        precisionRecallChart.update();
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('status').textContent = 'Error loading data';
                });
        }
        
        function toggleAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            if (checkbox.checked) {
                autoRefreshInterval = setInterval(updateDashboard, 2000);
            } else {
                clearInterval(autoRefreshInterval);
            }
        }
        
        // Initial load and start auto-refresh
        updateDashboard();
        toggleAutoRefresh();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the dashboard"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    """API endpoint to get training data"""
    metrics = load_yolov8_metrics()
    
    if not metrics:
        return jsonify({
            'error': 'No training data found. Make sure training has started and results.csv exists.',
            'epoch': 0,
            'last_update': datetime.now().strftime('%H:%M:%S')
        })
    
    current_epoch = len(metrics['epochs'])
    
    return jsonify({
        'epoch': current_epoch,
        'last_update': datetime.now().strftime('%H:%M:%S'),
        'train_loss': metrics['train_loss'][-1] if metrics['train_loss'] else None,
        'val_loss': metrics['val_loss'][-1] if metrics['val_loss'] else None,
        'map50': metrics['map50'][-1] if metrics['map50'] else None,
        'map': metrics['map'][-1] if metrics['map'] else None,
        'precision': metrics['precision'][-1] if metrics['precision'] else None,
        'recall': metrics['recall'][-1] if metrics['recall'] else None,
        'epochs': metrics['epochs'],
        'train_losses': metrics['train_loss'],
        'val_losses': metrics['val_loss'],
        'map50_values': metrics['map50'],
        'map_values': metrics['map'],
        'precision_values': metrics['precision'],
        'recall_values': metrics['recall']
    })

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv8 Training Dashboard')
    parser.add_argument('--results_dir', type=str, default='../data/models/yolov8_results/bark_classifier',
                        help='Directory containing YOLOv8 results (default: ./yolov8_results/bark_classifier)')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the web server on (default: 5000)')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to bind to (default: 127.0.0.1)')
    
    args = parser.parse_args()
    
    global RESULTS_DIR
    RESULTS_DIR = Path(args.results_dir)
    
    print(f"YOLOv8 Training Dashboard")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Server running on http://{args.host}:{args.port}")
    print("=" * 70)
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()

