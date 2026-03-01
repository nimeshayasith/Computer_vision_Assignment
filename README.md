# 🧠 Image Processing & Computer Vision — Assessment 01
### University of Ruhuna | Department of Electrical and Information Engineering
### Module: EE7204 / EC7205 | Student: Kumara K.N.Y. | Index: EG/2021/4624

---

> This repository contains my complete solutions for the **Image Processing and Computer Vision Take-Home Assessment**. Each question explores a core concept in classical image processing — implemented from scratch in Python (OpenCV + NumPy) without any AI or deep learning. If you are learning image processing, this repo is a practical, hands-on reference for 10 fundamental topics plus a real-world fundus image segmentation project.

---

## 📁 Repository Structure

```
├── Dataset/
│   └── IPCV_ASSIGNMENT_01_DATABASE/
│       ├── Images_For_Preliminary_Part/
│       │   ├── Image_1.jpg   ← Used in Q1 (Average Filtering)
│       │   ├── Image_2.jpg   ← Used in Q2 (Median Filtering / Noise)
│       │   ├── Image_3.jpg   ← Used in Q3, Q4, Q6, Q7 (Gaussian, Pyramids, Wavelet, Watermarking)
│       │   ├── Image_4.jpg   ← Used in Q8 (CT Organ Segmentation)
│       │   ├── Image_5.jpg   ← Used in Q9 (MRI Enhancement)
│       │   └── Image_6.jpg   ← Used in Q10 (Morphological Analysis)
│       └── Database_For_Practical_Part/
│           ├── Fundus Images For Validation/
│           └── Ground Truth For Validation/
├── practise/
│   ├── img1_original.png
│   ├── img1_result.png
│   ├── img1_gt.png
│   └── ... (10 test image triplets)
├── Question_01.py
├── Question_02.py
├── Question_03.py
├── Question_04.py
├── Question_05.py
├── Question_06.py
├── Question_07.py
├── Question_08.py
├── Question_09.py
├── Question_10.py
├── Practical_Work.py
└── README.md
```

---

## 🔧 Requirements

```bash
pip install opencv-python numpy matplotlib PyWavelets
```

All scripts were developed and tested on **Google Colab**. To run locally, simply update the image paths at the top of each script.

---

## 📚 Question-by-Question Concept Guide

---

### ❓ Question 01 — Average (Mean) Filtering
**File:** `Question_01.py`

#### 🔍 What is Average Filtering?
Average filtering is one of the simplest **spatial domain** smoothing techniques. It replaces each pixel with the **arithmetic mean** of all pixels inside a sliding window (kernel) centred on that pixel.

The mathematical operation is a **2D convolution** with a uniform kernel:

```
K(x, y) = 1 / (N × N)   for all positions in an N×N kernel
```

#### 🛠️ What was implemented?
- A **fully custom** average filter built using nested Python loops (no `cv2.blur`)
- Applied kernel sizes: **3×3, 5×5, 11×11, 15×15**
- Used **reflect padding** to handle image borders correctly

#### 📊 Key Observation
| Kernel Size | Effect |
|---|---|
| 3×3 | Mild smoothing, edges preserved |
| 5×5 | Moderate blur |
| 11×11 | Heavy blur, fine details lost |
| 15×15 | Very heavy blur, edges significantly softened |

> **Concept to remember:** Larger kernels → more neighbours averaged → stronger smoothing → more edge degradation. This is because high-frequency components (edges, textures) are progressively suppressed.

---

### ❓ Question 02 — Salt & Pepper Noise + Median Filtering
**File:** `Question_02.py`

#### 🔍 What is Salt & Pepper Noise?
Salt & pepper noise randomly sets pixels to either **pure white (255)** or **pure black (0)**, simulating sensor defects or transmission errors.

#### 🔍 What is Median Filtering?
Instead of averaging, the median filter sorts all pixel values inside the kernel window and picks the **middle value**. This makes it **resistant to outliers** — salt (255) and pepper (0) values are simply sorted to the extremes and never selected as the median.

#### 🛠️ What was implemented?
- Custom **salt & pepper noise** generator (10% and 20% corruption)
- Custom **median filter** using nested loops and `np.median()`
- Applied kernel sizes: **3×3, 5×5, 11×11** on the 20% noisy image

#### 📊 Key Observation
| Kernel Size | Noise Removal | Edge Preservation |
|---|---|---|
| 3×3 | Good for isolated pixels | Excellent |
| 5×5 | Good for moderate clusters | Good |
| 11×11 | Excellent for dense noise | Moderate blurring begins |

> **Why median over average for S&P noise?** The average filter pulls the output towards the extreme 0 or 255 values and spreads the noise. The median completely ignores these outliers.

---

### ❓ Question 03 — Gaussian Filtering
**File:** `Question_03.py`

#### 🔍 What is Gaussian Filtering?
Gaussian filtering applies a **bell-curve shaped kernel** to smooth an image. Unlike the uniform average filter, pixels closer to the centre contribute more than those further away. The 2D Gaussian function is:

```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)
```

Where **σ (sigma)** controls the width of the bell curve.

#### 🛠️ What was implemented?
- Custom **2D Gaussian kernel generator** from the mathematical formula
- Custom convolution engine applied to the kernel
- **Experiment 1:** Varying kernel size with fixed σ = 1.5 → sizes: 3×3, 5×5, 11×11, 15×15
- **Experiment 2:** Varying σ with fixed 15×15 kernel → σ ∈ {0.5, 2.0, 5.0}

#### 📊 Key Observation
| Parameter Changed | Effect |
|---|---|
| Larger kernel (fixed σ) | Minimal change once kernel > 3σ in width |
| Larger σ (fixed kernel) | Dramatically more blurring — σ is the true control |

> **Concept to remember:** The kernel size just determines the spatial extent of the filter. The **σ value** is what truly controls the amount of smoothing.

---

### ❓ Question 04 — Gaussian & Laplacian Pyramids
**File:** `Question_04.py`

#### 🔍 What is an Image Pyramid?
An image pyramid is a **multi-scale representation** of an image — a stack of images at progressively lower resolutions. They are heavily used in computer vision tasks like object detection, image blending, and optical flow.

#### 🔍 Gaussian Pyramid
Each level is created by:
1. Blurring with a **5×5 Gaussian kernel** (anti-aliasing)
2. **Downsampling** by a factor of 2 (`cv2.pyrDown`)

#### 🔍 Laplacian Pyramid
Each level captures the **detail lost** between two Gaussian levels:
```
L_i = G_i − expand(G_{i+1})
```
Where `expand` upsamples `G_{i+1}` back to the size of `G_i`. The result highlights **edges and fine textures** at each scale.

#### 🛠️ What was implemented?
- 3-level Gaussian pyramid
- 3-level Laplacian pyramid with contrast boosting for visualisation (`L×4 + 128`)

> **Why 5×5 kernel?** It provides enough anti-aliasing before downsampling to prevent aliasing artefacts while being computationally cheap.

---

### ❓ Question 05 — Custom vs Built-in Filter Comparison
**File:** `Question_05.py`

#### 🔍 What is this about?
This question validates the correctness of the custom implementations from Q1, Q2, and Q3 by comparing them against OpenCV's highly optimised built-in functions.

#### 🛠️ What was implemented?
- For each filter (Average, Median, Gaussian) at the largest kernel:
  - `A` = custom implementation output
  - `B` = OpenCV built-in output
  - Computed: `diff = |A - B|`
  - Enhanced: `diff × 10` using a hot colourmap to reveal sub-pixel errors

#### 📊 Results
| Filter | Difference | Reason |
|---|---|---|
| Average | ~0 | Identical uniform kernel math |
| Median | ~0 | Same median selection algorithm |
| Gaussian | Tiny | OpenCV uses separable approximation; custom uses full 2D formula |

> **Key insight:** Even tiny floating-point differences become visible under the ×10 enhancement, confirming the implementations are mathematically equivalent.

---

### ❓ Question 06 — Wavelet Denoising
**File:** `Question_06.py`

#### 🔍 What is the Discrete Wavelet Transform (DWT)?
The DWT decomposes an image into **multi-frequency subbands**:
- **cA** — Approximation (low frequency, smooth structure)
- **cH** — Horizontal details (high frequency)
- **cV** — Vertical details (high frequency)
- **cD** — Diagonal details (high frequency)

#### 🛠️ What was implemented?
**Step 1 — Degradation:**
```
I' = I + SP + L(I)
```
Where:
- `SP` = 5% salt & pepper noise
- `L(I)` = Laplacian-filtered image (amplifies edges and ringing)

**Step 2 — Denoising via Haar DWT:**
1. Apply `pywt.dwt2(I', 'haar')` → get cA, cH, cV, cD
2. Zero out all detail bands: `cH = cV = cD = 0`
3. Reconstruct: `pywt.idwt2((cA, (0, 0, 0)), 'haar')`

#### 📊 Key Insight
> Zeroing the detail bands acts as a **low-pass filter in the wavelet domain**, perfectly suppressing noise and edge artefacts while retaining the smooth global structure from `cA`.

---

### ❓ Question 07 — DWT Digital Watermarking
**File:** `Question_07.py`

#### 🔍 What is Digital Watermarking?
Digital watermarking **hides ownership information** invisibly inside an image. A good watermark should be:
- **Imperceptible** — invisible to the human eye
- **Robust** — survives common image processing operations
- **Extractable** — recoverable by the owner

#### 🛠️ Embedding Process
```
1. Apply Haar DWT to cover image → get (cA, cH, cV, cD)
2. Embed: cA_watermarked = cA + α × W_normalised
3. Apply inverse DWT (IDWT) → watermarked image
```
Where `α = 15.0` controls the strength trade-off.

#### 🛠️ Extraction Process
```
1. Apply DWT to watermarked image → get cA_w
2. Apply DWT to original cover image → get cA
3. Recover: W = (cA_w − cA) / α
```

#### 📊 Why embed in the cA band?
> The **approximation band** carries most of the image energy and survives compression and filtering. This makes the watermark robust while remaining invisible since changes to cA affect low-frequency content that the human eye is less sensitive to at small amplitudes.

---

### ❓ Question 08 — CT Organ Segmentation
**File:** `Question_08.py`

#### 🔍 What is Image Segmentation?
Segmentation partitions an image into meaningful regions. In medical imaging, this means isolating specific anatomical structures from CT or MRI scans.

#### 🛠️ What was implemented?
A **colour-guided HSV masking pipeline**:
1. Load the coloured reference organ map alongside the CT scan
2. Resize reference map to match CT dimensions
3. Convert to **HSV colour space** for robust colour isolation
4. Define HSV ranges per organ:

| Organ | Colour in Map | HSV Range |
|---|---|---|
| Liver | Red | H: 0–10 and 170–179 |
| Right Kidney | Purple | H: 125–140 |
| Left Kidney | Magenta | H: 145–160 |
| Spleen | Blue | H: 110–125 |
| Spinal Column | Pink | H: 160–175 |

5. Apply `cv2.inRange()` to extract binary masks
6. Filter small contours (area < 200px) to remove text labels
7. Apply masks via `cv2.bitwise_and` on the CT image

> **Why HSV over BGR/RGB for colour detection?** HSV separates **hue** (colour type) from **saturation** and **value** (brightness), making colour detection far more robust to lighting variations.

---

### ❓ Question 09 — MRI Image Enhancement
**File:** `Question_09.py`

#### 🔍 The three enhancement techniques

**1. Histogram Equalisation (`cv2.equalizeHist`)**
Redistributes pixel intensities to achieve a **flat (uniform) histogram**, maximising global contrast. Effective for low-contrast images but can amplify noise.

**2. Contrast Stretching (`cv2.normalize`)**
Linearly maps the existing intensity range [min, max] to the full [0, 255] range. A gentler enhancement that **preserves relative tonal relationships**.

**3. Gaussian Filtering (`cv2.GaussianBlur`)**
Reduces high-frequency noise by smoothing. Does **not** enhance contrast — included as a baseline for noise reduction.

#### 📊 Clinical Diagnosis Recommendation
| Method | Best For | Risk |
|---|---|---|
| Histogram EQ | Very low contrast images | Over-enhancement, noise amplification |
| Contrast Stretching ✅ | Moderately narrow histograms | None significant |
| Gaussian Filtering | Noisy images only | Blurs fine anatomical detail |

> **Recommended for clinical use: Contrast Stretching** — it improves tissue differentiation without introducing artefacts that could be mistaken for pathology.

---

### ❓ Question 10 — Morphological Image Analysis
**File:** `Question_10.py`

#### 🔍 What is Mathematical Morphology?
Morphological operations process images based on **shape**. They use a **structuring element** (kernel) that probes the image geometry.

#### 🔍 The four operations (5×5 rectangular kernel)

| Operation | Formula | Effect |
|---|---|---|
| **Erosion** | `A ⊖ B` | Shrinks objects, removes small noise blobs |
| **Dilation** | `A ⊕ B` | Expands objects, fills small holes |
| **Opening** | `(A ⊖ B) ⊕ B` | Erosion then dilation — removes external noise |
| **Closing** | `(A ⊕ B) ⊖ B` | Dilation then erosion — fills internal gaps |

#### 🛠️ What was implemented?
1. **Otsu's thresholding** — automatic binary conversion
2. All four morphological operations via OpenCV
3. **Feature extraction** via contour analysis on the closed image:
   - `cv2.contourArea()` → Area in pixels
   - `cv2.arcLength()` → Perimeter in pixels

> **Concept to remember:** Opening and Closing are inverses of each other. Use **Opening** to clean the outside of shapes; use **Closing** to clean the inside.

---

## 🏥 Practical Work — Retinal Blood Vessel Segmentation
**File:** `Practical_Work.py`

### 🎯 Objective
Segment the **retinal blood vessel network** from fundus images using a fully classical (non-AI) image processing pipeline, and validate against 100 ground truth masks.

### 🔬 Pipeline Overview

```
RGB Fundus Image
      ↓
1. Resize to 800×800 + Green Channel Extraction
      ↓
2. FOV Masking (removes black background artefacts)
      ↓
3. First CLAHE (corrects uneven global illumination)
      ↓
4. Background Subtraction (45×45 Median Blur)
      ↓
5. Double-CLAHE (amplifies faint capillary signals)
      ↓
6. Statistical Thresholding: μ + 1.2σ (adaptive per image)
      ↓
7. Morphological Closing + Connected Component Analysis
      ↓
Binary Blood Vessel Mask
```

### 🔍 Why each step?

| Step | Reason |
|---|---|
| Green channel | Maximises vessel-to-background contrast in fundus images |
| FOV mask | Prevents bright border from corrupting statistics |
| First CLAHE | Equalises uneven fundus illumination before processing |
| Background subtraction | Median blur at 45px erases vessels; subtracting reveals them |
| Double-CLAHE | Second CLAHE dramatically boosts faint capillaries post-subtraction |
| μ + 1.2σ threshold | Adapts to each image's unique brightness profile |
| CCA (area > 20) | Removes isolated noise pixels without affecting vessel network |

### 📊 Validation Results

| Metric | Score (100 images) |
|---|---|
| Dice Similarity Coefficient (DSC) | **0.6828** |
| Jaccard Index (IoU) | **0.5258** |

> Achieving **~68% DSC without any deep learning** demonstrates the effectiveness of well-designed classical pipelines. For reference, state-of-the-art deep learning methods achieve ~82% DSC on similar datasets.

### 🔍 Validation Metrics Explained

**Dice Similarity Coefficient (DSC):**
```
DSC = 2|S ∩ G| / (|S| + |G|)
```
Measures overlap between predicted mask `S` and ground truth `G`. Ranges 0→1.

**Jaccard Index (IoU):**
```
IoU = |S ∩ G| / |S ∪ G|
```
Ratio of intersection to union. More strict than DSC. Ranges 0→1.

---

## 💡 Key Concepts Summary

| Topic | Q | Core Idea |
|---|---|---|
| Average Filter | 1 | Uniform kernel convolution — larger kernel = more blur |
| Median Filter | 2 | Rank filter — immune to salt & pepper outliers |
| Gaussian Filter | 3 | Weighted smoothing — σ controls spread |
| Image Pyramids | 4 | Multi-scale representation — Gaussian ↓, Laplacian = detail |
| Filter Validation | 5 | Custom ≈ built-in; differences are floating-point rounding only |
| Wavelet Denoising | 6 | Zero detail bands in DWT domain = low-pass filtering |
| DWT Watermarking | 7 | Embed in cA band for robustness + imperceptibility |
| CT Segmentation | 8 | HSV masking on reference colour map |
| MRI Enhancement | 9 | Contrast stretching best for clinical use |
| Morphology | 10 | Shape-based operations: open=clean outside, close=clean inside |
| Fundus Segmentation | P | Double-CLAHE + adaptive thresholding → 68% DSC |

---

## 🚀 How to Run

**On Google Colab (recommended):**
```python
from google.colab import drive
drive.mount('/content/drive')
# Then run any Question_XX.py file directly
```

**Locally:**
```bash
git clone https://github.com/nimeshayasith/Computer_vision_Assignment
cd Computer_vision_Assignment
pip install opencv-python numpy matplotlib PyWavelets
python Question_01.py
```

---

## 📖 References

- Gonzalez, R.C. & Woods, R.E. — *Digital Image Processing*, 4th Edition
- OpenCV Documentation — https://docs.opencv.org
- PyWavelets Documentation — https://pywavelets.readthedocs.io
- Mallat, S. — *A Wavelet Tour of Signal Processing*

---

## 👤 Author

**Kumara K.N.Y.**
- Index: EG/2021/4624
- Programme: Computer Engineering (CE)
- University of Ruhuna, Sri Lanka
- GitHub: [nimeshayasith](https://github.com/nimeshayasith/Computer_vision_Assignment)

---

*If you found this helpful for learning image processing, feel free to ⭐ star the repository!*
