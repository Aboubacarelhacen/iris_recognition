# Iris Recognition System (Daugman's IrisCode)

A complete iris recognition pipeline implemented in a single Jupyter Notebook,
following the classic approach proposed by John Daugman.

## Pipeline

1. **Segmentation** — detect the pupil and iris boundary using the Hough Transform.
2. **Normalization** — unwrap the iris ring into a 64 × 512 rectangle (Daugman's rubber sheet model).
3. **Feature Extraction** — encode texture with a bank of 2-D Gabor filters into a 2048-bit IrisCode, with eyelid occlusion masking.
4. **Matching** — compare IrisCodes using rotation-tolerant masked Hamming Distance.
5. **Evaluation** — compute FAR, FRR, EER and ROC/AUC over genuine vs impostor pairs.

## Dataset

This project uses the **CASIA-Iris-Thousand** dataset (1000 subjects, ~20,000 images).
Available on [Kaggle](https://www.kaggle.com/).

After downloading, place or symlink the dataset so the folder structure looks like:

```
data/CASIA-IrisV1/
├── 000/
│   ├── L/
│   │   ├── S5000L00.jpg
│   │   └── ...
│   └── R/
│       ├── S5000R00.jpg
│       └── ...
├── 001/
│   └── ...
└── ...
```

The notebook auto-detects the L/R subfolder layout.

## Installation

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

## Usage

```bash
jupyter notebook iris_recognition.ipynb
```

Run all cells top to bottom (`Kernel → Restart & Run All`).
Set `MAX_SUBJECTS = None` in the dataset cell to process the full 20,000-image dataset (~45 min),
or keep it at `50` for a quick run (~2 min).

## Method

| Step | Technique |
|---|---|
| Segmentation | `cv2.HoughCircles` for pupil and iris boundary |
| Normalization | Polar unwrap to 64 × 512 (rubber sheet) |
| Occlusion mask | Eyelid rows excluded from encoding |
| Encoding | 2-D Gabor filters (imaginary part) at 4 orientations → 2048-bit IrisCode |
| Matching | Rotation-tolerant masked Hamming Distance (±8 column shifts) |
| Evaluation | FAR/FRR sweep, EER, ROC + AUC |

## Output

- Segmentation overlay (pupil and iris circles on the eye image)
- Normalized iris strip (64 × 512)
- IrisCode visualized as a binary heatmap
- Genuine vs impostor Hamming distance histograms
- FAR/FRR curves with EER point
- ROC curve with AUC
- Results summary table
