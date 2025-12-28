# Prior-information-aided three-dimensional fatigue delamination prognostics in composites

Data and code for the paper **“Prior information-aided three-dimensional fatigue delamination prognostics in composites.”**

## Installation
### Requirements
- Python ≥ 3.8
- CUDA (optional, 11.x recommended)
- PyTorch ≥ 1.10
- NumPy, OpenCV, Matplotlib

## Notices
- **Raw data.** `APPENDIXB1`, `APPENDIXB2`, `APPENDIXB3`, `APPENDIXB3`, `APPENDIXB5` are the **original ultrasonic C-scan images** for each specimen.
- **Preprocess.** `feature_map+reconstruction/` converts C-scans to 5-channel feature maps and provides simple visualization.
- **Prognostics code.** `prognostics/` contains:
  - `U-Tprognostic.py`: main training/evaluation script (U-Net + temporal module with prior-informed losses).
  - `run.sh`: example runner.
  - `data/`, `data_110/`, `data_220/`, `data_330/`, `data_440/`: preprocessed dataset variants.

## Data Availability
- The original C-scan data are described in the paper.
- Preprocessed **5-channel feature-map tensors** are provided in .npy format.

## Citation
If you find this repo or our work useful, please consider citing:

- **Prior information-aided three-dimensional fatigue delamination prognostics in composites.**
- **Particle filter-based delamination shape prediction in composites subjected to fatigue loading.**
- **Multiple local particle filter for high-dimensional system identification.**

For collaboration or extended features (e.g., image processing/visualization tools or research on PHM/RUL prediction), please contact the author at yuanjing930418@gmail.com
