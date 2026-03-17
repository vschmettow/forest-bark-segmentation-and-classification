# Configuration

- **docs/** – Training guides, Colab notebooks, installation
- **dataset.yaml** – Generated per dataset (BarkNetYOLO/stage1_80/dataset.yaml, etc.)

Example dataset.yaml:
```yaml
path: /path/to/BarkNetYOLO/stage1_80
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
```
