# Dataset

## BarkNet / PlantNet RGB

Species-level bark images for *Picea abies* (Norway spruce) and *Pinus sylvestris* (Scots pine).

- **Total:** 1,931 images
- **Split:** 1,537 train, 330 val, 64 test (70/15/15)
- **Classes:** Picea (0), Pinus (1)
- **Format:** YOLO segmentation (polygon masks), LabelMe JSON (source)

## Folder Structure (not in repo – too large)

| Folder | Description |
|--------|-------------|
| `rawBarkNet/` | Source data: LabelMe JSON + images, species folders |
| `BarkNetYOLO/` | YOLO format: stage1_80 … stage4_363, test_set, final_dataset |
| `BarkNetCOCO/` | COCO format (for ConvexMask) |

## Creating the Dataset

1. Place raw data in `rawBarkNet/` (LabelMe format).
2. Run conversion:
   ```bash
   python code/scripts/convert_barknet_to_coco_and_yolo.py
   python code/scripts/add_data4_and_test_set.py
   python code/scripts/create_staged_datasets.py
   python code/scripts/create_final_dataset.py
   ```

## Staged Datasets

| Stage | Images | Picea | Pinus |
|-------|--------|-------|-------|
| stage1_80 | 80 | 35 | 45 |
| stage2_160 | 160 | 77 | 83 |
| stage3_280 | 280 | 136 | 144 |
| stage4_363 | 363 | 181 | 181 |
