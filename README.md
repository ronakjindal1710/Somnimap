# SomniMap

Explainable Multi-Task Deep Learning for Sleep Architecture Mapping on the PhysioNet CAP Sleep Database.

## Setup
- Raw dataset (read-only): `C:/Users/prave/OneDrive/Desktop/Somnimap/dataset`
- Project root: `C:/Users/prave/.gemini/antigravity/scratch/SomniMap`
- Base config: `experiments/configs/base_config.yaml`

## Phase 0 Scripts
1. `python src/data/inventory.py` -> `data/manifest.csv`
2. `python src/data/annotation_parser.py` -> `data/annotation_coverage.csv`
3. `python src/data/split_manifest.py` -> `data/splits/split_manifest.json`
