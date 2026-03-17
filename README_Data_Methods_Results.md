# Forest Instance Segmentation (Bark/Tree Stems) – Data, Methods & Results

**Project Goal:** Species-level detection of *Picea abies* and *Pinus sylvestris*  
**Primary Model:** YOLOv8s-seg  
**Dataset:** PlantNet RGB dataset (n=1,931) with balanced classes  
**Evaluation Metrics:** Precision, Recall, F1, mAP50, mAP50-95  

**GitHub Link:** [https://github.com/vschmettow/forest-bark-segmentation-and-classification](https://github.com/vschmettow/forest-bark-segmentation-and-classification)

---

## 1. Dataset & Pre-processing (Detailed)

### Image counts


| Split                    | Count | Notes                                   |
| ------------------------ | ----- | --------------------------------------- |
| **Training**             | 1,537 | 70/85 of trainable data (test excluded) |
| **Validation**           | 330   | 15/85 of trainable data                 |
| **Test**                 | 64    | Fixed hold-out set (33 Picea, 31 Pinus) |
| **Total (trainable)**    | 1,867 | Excludes test set                       |
| **Total (full dataset)** | 1,931 | Train + val + test                      |


- Split ratios: 70% train, 15% val, 15% test (test excluded from training; of non-test: train = 70/85, val = 15/85).
- Stratified split by species; random seed 42.

### Species counts (full dataset, excluding test)


| Species            | Count |
| ------------------ | ----- |
| *Picea abies*      | 932   |
| *Pinus sylvestris* | 935   |


### Species counts per experimental stage

Stages are cumulative subsets from `data_all` (363 images), stratified by Picea/Pinus ratio:


| Stage      | Total images | Train (80%) | Val (20%) | Picea | Pinus |
| ---------- | ------------ | ----------- | --------- | ----- | ----- |
| stage1_80  | 80           | 64          | 16        | X     | X     |
| stage2_160 | 160          | 128         | 32        | X     | X     |
| stage3_280 | 280          | 224         | 56        | X     | X     |
| stage4_363 | 363          | 290         | 73        | X     | X     |


*Exact per-species counts per stage: stratified sampling from pool; ratio preserved from `data_all`.*

### Pre-processing and data pipeline

- **Source:** `rawBarkNet` (LabelMe JSON annotations; species folders `Picea abies L. H.Karst`, `Pinus sylvestris L`).
- **Conversion:** `convert_barknet_to_yolo.py` → BarkNetYOLO (70/15/15).
- **Test set:** `add_data4_and_test_set.py` creates fixed 64-image test set (33 Picea, 31 Pinus).
- **Staged datasets:** `create_staged_datasets.py` builds stage1_80 … stage4_363 from `data_all` with 80/20 train/val.
- **Final dataset:** `create_final_dataset.py` builds `final_dataset` from data0–5 + root species folders, excluding test set.
- **Annotation tool:** LabelMe (private labeling tool); AnyLabeling used for labeling (see `LabellingTool/`). Any bad quality images removed. Labeled in batches, then labeling tool trained, it prelabeled and I manually correected the labels. Then trained again, etc etc. Optimised workflow
- **Data cleaning:** Deduplication by stem ID (first occurrence kept); `remove_right_folders_from_flat.py` for FinnWoodlands flattening.

---

## 2. Experimental Methods (Successful Path)

### Hardware and environment

- **Primary:** Google Colab, T4 GPU (16 GB).
- **Image size:** 640×640 (`imgsz=640`).
- **Framework:** Ultralytics YOLOv8; `yolov8s-seg.pt` as pretrained backbone.
- **Reference:** X (Colab runtime specs).

### Four data-size stages (80, 160, 280, 363 images)


| Stage      | Train | Val | Epochs | Batch | Notes           |
| ---------- | ----- | --- | ------ | ----- | --------------- |
| stage1_80  | 64    | 16  | 80     | 4     | Smallest        |
| stage2_160 | 128   | 32  | 80     | 4     |                 |
| stage3_280 | 224   | 56  | 80     | 4     |                 |
| stage4_363 | 290   | 73  | 80     | 4     | Full staged set |


- Cumulative: stage1 ⊂ stage2 ⊂ stage3 ⊂ stage4.
- All stages: batch=4, epochs=80, imgsz=640.

### Stage 2 batch vs. epoch experiments

Same dataset: `BarkNetYOLO/stage2_160` (128 train, 32 val).


| Experiment | Batch | Epochs | Model path                                            |
| ---------- | ----- | ------ | ----------------------------------------------------- |
| Base       | 4     | 80     | `best_models/yolov8s_seg_stage2_160_best.pt`          |
| b16_e100   | 16    | 100    | `best_models/yolov8s_seg_stage2_160_b16_e100_best.pt` |
| b32_e100   | 32    | 100    | `best_models/yolov8s_seg_stage2_160_b32_e100_best.pt` |


### Final dataset training

- **Data:** `BarkNetYOLO/final_dataset` (1,537 train, 330 val; 932 Picea, 935 Pinus, test excluded).
- **Settings:** epochs=100, batch=32, imgsz=640.
- **Model:** `best_models/yolov8s_seg_final_dataset_best.pt`.

---

## 3. Failed Methods & Experimental "Dead-Ends"

- **Segment-then-crop-and-classify pipeline:** Two-stage pipeline (SAM2/SAM1 segmentation → crop → YOLOv8 classifier) implemented in `integrate_segmentation_with_classifier.py` and `forest_bark_analyzer.py`. Not used as primary approach; end-to-end YOLOv8s-seg preferred. Failed because even though SAM uses language as an input for segmentation (e.g. "Bark") it doe not segment very well. The classification of the trees also did not suffice since the zooming in on the bark in a forest scenery image would make the image quality too bad so that nothing could be classified well enough.
- **FinnWoodlands dataset:** Used in `forest_instance_segmentation/`; separate pipeline with COCO→YOLO conversion. Not part of main BarkNet/PlantNet experiments. Quickly realised that the images are too low quality and the predone masks were not specific enough
- **ConvexMask R50-FPN:** Trained on BarkNetCOCO/data0; documented in `TRAINING_BARKNET.md` and `train_barknet_convexmask.py`. Alternative model; YOLOv8s-seg chosen as primary. ConvexMask was difficult to implement, did not find. thereasons and had to prioritise. Convexmask is meant to be used in more complex structures, better for forest scenery images
- **YOLOv8 classifier (standalone):** Trained on cropped bark images (600×600); used in segment-then-classify pipeline. Superseded by end-to-end YOLOv8s-seg. this was used for the segment then crop and classify pipeline. WOrked ok on the bark, but not on the zoomed / cropped images
- **SAM2/SAM1 for segmentation:** Used in web apps for interactive segmentation; not used for training or main evaluation. Used in segment then crop and classify pieline but the segmentation based on language simply did not work as well as a specifically trained model could

---

## 4. Results & Best Model Analysis

### Metrics (where available)

From `evaluation_report/` (YOLOv8s-seg on data0 and data_all, validated on respective val/test):


| Model                  | mAP50  | mAP50-95 | Precision | Recall | F1    |
| ---------------------- | ------ | -------- | --------- | ------ | ----- |
| YOLOv8s-seg (data0)    | 0.6722 | 0.4408   | 0.9054    | 0.5816 | ~0.71 |
| YOLOv8s-seg (data_all) | 0.8269 | 0.7761   | 0.9153    | 0.7673 | ~0.83 |


### Staged experiments (stage1–4, batch=4, epochs=80)

*Validation on test set (64 images).*


| Stage      | mAP50  | mAP50-95 | Precision | Recall | F1    |
| ---------- | ------ | -------- | --------- | ------ | ----- |
| stage1_80  | 0.7977 | 0.7396  | 0.7472    | 0.7653 | 0.7561 |
| stage2_160 | 0.8448 | 0.8007  | 0.7779    | 0.8053 | 0.7914 |
| stage3_280 | 0.8836 | 0.8361  | 0.9519    | 0.8071 | 0.8735 |
| stage4_363 | 0.8834 | 0.8290  | 0.9461    | 0.8468 | 0.8937 |


**Best stage model path:** `best_models/yolov8s_seg_stage4_363_best.pt` (highest F1 among stages).

### Stage 2 batch/epoch comparison


| Model                             | mAP50  | mAP50-95 | Precision | Recall | F1    |
| --------------------------------- | ------ | -------- | --------- | ------ | ----- |
| stage2_160 (batch=4, epochs=80)   | 0.8448 | 0.8007  | 0.7779    | 0.8053 | 0.7914 |
| stage2_160 (batch=16, epochs=100) | 0.7805 | 0.7363  | 0.8619    | 0.7323 | 0.7918 |
| stage2_160 (batch=32, epochs=100) | 0.8356 | 0.7980  | 0.9064    | 0.7599 | 0.8267 |


**Best stage2 model path:** `best_models/yolov8s_seg_stage2_160_best.pt` (highest mAP50); `best_models/yolov8s_seg_stage2_160_b32_e100_best.pt` (highest F1, best precision).

### Final dataset model


| Model                                | mAP50  | mAP50-95 | Precision | Recall | F1    |
| ------------------------------------ | ------ | -------- | --------- | ------ | ----- |
| final_dataset (batch=32, epochs=100) | 0.9029 | 0.8730  | 0.9365    | 0.8925 | 0.9140 |


**Best full-dataset model path:** `best_models/yolov8s_seg_final_dataset_best.pt` (best overall).

### Interpretation (where applicable)

- **Batch size 4 as regularization:** For stage2_160, batch=4 (mAP50 0.8448) outperforms batch=16 (0.7805) and batch=32 (0.8356) on mAP50. The smaller batch yields noisier gradients that may improve generalization on limited data. Batch=32 achieves highest precision (0.906) and F1 (0.827), trading some recall for fewer false positives.
- **Performance saturation at 280 images:** mAP50 plateaus from stage3 (0.8836) to stage4 (0.8834); stage4 gains in recall (0.847 vs 0.807) and F1 (0.894 vs 0.874). Diminishing returns suggest ~280 images may be sufficient for this task, though more data still helps recall.
- **data_all vs data0:** data_all (more data) shows clear gains: mAP50 0.67→0.83, mAP50-95 0.44→0.78, recall 0.58→0.77. More data improves both detection and segmentation quality.

---

## 5. References (Placeholders)

- Condat et al. (2024) – ConvexMask, tree instance segmentation. *IEEE Robotics and Automation Letters*. X
- PlantNet dataset: X
- YOLOv8/Ultralytics: X
- LabelMe: X

