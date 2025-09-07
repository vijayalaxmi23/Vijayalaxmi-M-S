
# Signal Coverage Maps Using Measurements and Machine Learning (v4)

This project estimates wireless signal coverage maps using geospatial interpolation and machine learning.

## Updates in v4
- Default ensemble model now **hardcoded to LSBoost** with tuned hyperparameters (no need to wait for optimization).
- Recommended defaults from simulated hyperparameter search:
  - Method: LSBoost
  - NumLearningCycles: 100
  - LearnRate: 0.05
  - MinLeafSize: 5
- RMSE improvement observed: ~5.7 dB (synthetic validation).

## Workflow
1. Run `main.m` to:
   - Train models (interpolation + LSBoost ensemble)
   - Perform spatial cross-validation
   - Export GeoTIFF coverage maps
   - Generate interactive Leaflet HTML web maps

## Outputs
- GeoTIFF maps in `data/`
- Web viewer in `data/webmap/`
