# Colab – YOLOv8s-seg Training

**Upload** `BarkNetYOLO` to `My Drive` / `Colab Notebooks` / `Bark` / `BarkNetYOLO/`

---

## All training (stages + stage2 comparison)

**[COLAB_YOLO_ALL_STAGES.md](COLAB_YOLO_ALL_STAGES.md)** – One notebook: stages 1–4 + stage2_160 comparison (baseline b4/e80 from Cell 5, b16/e100 from Cell 8, b32/e100 from Cell 9). Mount Drive once, run cells in order.

For stage2_160 comparison HTML, run locally:
```bash
python scripts/generate_stage2_comparison_report.py
```
Output: `Results/stage2_comparison.html`

---

## After training

1. Download each best model to `best_models/`:
   - `yolov8s_seg_stage1_80_best.pt` … `stage4_363_best.pt`
   - `yolov8s_seg_stage2_160_best.pt` (Cell 5), `yolov8s_seg_stage2_160_b16_e100_best.pt` (Cell 8), `yolov8s_seg_stage2_160_b32_e100_best.pt` (Cell 9)

2. Generate HTML reports:
   ```bash
   python scripts/generate_stages_combined_report.py
   ```
   Output: `Results/stages_combined_report.html`
   ```bash
   python scripts/generate_stage2_comparison_report.py
   ```
   Output: `Results/stage2_comparison.html`
