# Large datasets (local only — not on GitHub)

Place large image/annotation datasets here. **Subfolders of `data/datasets/` are ignored by Git** (see root `.gitignore`), so they are never pushed to GitHub.

## Examples

- **FinnWoodlands**: Can live at repo root as `FinnWoodlands/` (also gitignored) or be moved/symlinked here as `data/datasets/FinnWoodlands/` for a single shared location.
- Any other heavy datasets (images, masks, etc.) should go in a subfolder of `data/datasets/` so they stay local.

## Paths in code

- From repo root: `data/datasets/<dataset_name>/`
- From `forest_instance_segmentation/`: `../data/datasets/<dataset_name>/` or `../../data/datasets/<dataset_name>/` depending on script location.
