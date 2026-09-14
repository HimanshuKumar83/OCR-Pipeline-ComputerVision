# Robust OCR Pipeline

An interview-friendly OCR application that compares multiple OpenCV preprocessing strategies and Tesseract layouts before choosing the strongest result. It includes both a command-line pipeline and an interactive Streamlit browser demo.

## What It Does

The project accepts a photograph, scan, sign, or document image and returns cleaned, editable text. Rather than applying one aggressive threshold to every image, it creates several conservative candidates and runs Tesseract on each one. Word-level confidence, recognized word count, and non-empty line count determine which result is displayed.

## Technologies

- **Python 3.10+:** application language and CLI.
- **OpenCV:** image loading, resizing, grayscale conversion, denoising, CLAHE, thresholding, and deskewing.
- **Tesseract OCR:** external OCR engine that recognizes the text.
- **pytesseract:** Python wrapper used to call Tesseract and read word-level confidence data.
- **Streamlit:** interactive browser interface.
- **NumPy:** converts uploaded browser image bytes into OpenCV image arrays.

## Features

- Image path validation and clear error messages.
- Upload an image or use the included sample in the browser demo.
- Conditional 2x upscaling for smaller images.
- Lightweight automatic deskewing when meaningful skew is detected.
- Original image candidate, preserving information before thresholding.
- Four preprocessing variants:
  - Minimal grayscale and light denoising.
  - CLAHE contrast enhancement without binarization.
  - Otsu global thresholding.
  - Adaptive Gaussian thresholding.
- Three Tesseract page segmentation modes: PSM 3, 6, and 11.
- Confidence-based selection using `pytesseract.image_to_data()`.
- Cleaned output that preserves meaningful line breaks.
- Optional saving of the winning preprocessed image and text output.
- Browser view of the best candidate, confidence, selected method, PSM, and every OCR attempt.

## Project Structure

```text
robust-ocr-pipeline/
├── app.py                         # Streamlit browser interface
├── ocr_pipeline.py                # OCR engine and command-line interface
├── requirements.txt               # Python dependencies
├── README.md
├── .gitignore
├── sample_input/
│   └── sample.png                 # Included test image
├── sample_output/
│   └── output.txt                 # Example OCR output
└── preprocessing/
    └── preprocessing_steps.png    # Preprocessing image asset
```

## How the Pipeline Works

```text
Input image
    |
    v
Upscale if small + deskew when needed
    |
    +--> Original image
    +--> Minimal grayscale
    +--> CLAHE contrast
    +--> Otsu threshold
    +--> Adaptive threshold
                 |
                 v
       Tesseract PSM 3, 6, 11
                 |
                 v
       image_to_data() confidence metrics
                 |
                 v
       Best result selection + cleaning
                 |
                 v
            Extracted text
```

Multiple strategies are more robust than one fixed threshold because image conditions vary. A clean scan may work best with the original or grayscale image, while uneven lighting may benefit from adaptive thresholding. The selector compares the candidates instead of assuming thresholding is always better.

## Installation

### 1. Open the project folder

In PowerShell:

```powershell
cd "C:\Users\hp\Desktop\COMPUTERVISIONPP\robust-ocr-pipeline"
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Tesseract is a separate system application, not a Python package.

**Windows with WinGet:**

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Close and reopen PowerShell or VS Code after installation, then verify:

```powershell
tesseract --version
```

If `tesseract` is not recognized, add this folder to your Windows User `Path` environment variable:

```text
C:\Program Files\Tesseract-OCR
```

Other operating systems:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install tesseract-ocr

# macOS with Homebrew
brew install tesseract
```

The project does not hardcode a machine-specific Tesseract path. It expects the `tesseract` command to be available on `PATH`.

## Run Locally: Command Line

Make sure the terminal is inside the project folder and the virtual environment is active.

Run the sample image:

```powershell
python ocr_pipeline.py --image sample_input\sample.png
```

Save the selected preprocessing image and extracted text:

```powershell
python ocr_pipeline.py `
  --image sample_input\sample.png `
  --save-preprocessed preprocessing\preprocessed.png `
  --output sample_output\output.txt
```

The default command compares 15 combinations: 5 image candidates multiplied by PSM 3, 6, and 11. To test one PSM mode only:

```powershell
python ocr_pipeline.py --image sample_input\sample.png --psm 6
```

The `--image` argument is required. Running only `python ocr_pipeline.py` displays the usage help and an expected missing-argument error.

## Run Locally: Browser Demo

From the project folder with `.venv` activated:

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually:

```text
http://localhost:8501
```

Open that address in a browser. The interface lets you:

1. Use the included sample or upload PNG, JPG, JPEG, or BMP images.
2. Click **Run OCR**.
3. View the original image and the selected best preprocessed image.
4. See extracted text in a readable output box.
5. Inspect confidence, preprocessing method, and PSM mode.
6. Expand **Show all OCR attempts** to compare every candidate.
7. Download the result as `ocr_output.txt`.

If the terminal is opened in the parent folder, use an absolute command to avoid importing the wrong file:

```powershell
& "C:\Users\hp\Desktop\COMPUTERVISIONPP\robust-ocr-pipeline\.venv\Scripts\streamlit.exe" run `
  "C:\Users\hp\Desktop\COMPUTERVISIONPP\robust-ocr-pipeline\app.py"
```

## Sample Result

Input file: `sample_input/sample.png`

Expected text:

```text
PathPal OCR Demo
Obstacle ahead - Distance: 12 meters
Crosswalk detected
```

The included sample was tested with the confidence-guided pipeline. It selected Adaptive threshold with PSM 11 at approximately 94.9% mean word confidence. Exact text and confidence can vary by Tesseract version and operating system.

## Important Functions

- `preprocess_minimal()` preserves character shapes with light processing.
- `preprocess_clahe()` improves local contrast without binarization.
- `preprocess_otsu()` handles clean, evenly lit images.
- `preprocess_adaptive()` handles uneven backgrounds as one alternative.
- `deskew_image()` rotates only when a meaningful angle is estimated.
- `run_ocr()` calls both `image_to_data()` for metrics and `image_to_string()` for readable text.
- `calculate_confidence()` computes mean confidence, word count, and line count.
- `select_best_result()` ranks the OCR attempts.
- `recognize_image()` runs the complete comparison.

## Troubleshooting

### `TesseractNotFoundError`

Install Tesseract, add `C:\Program Files\Tesseract-OCR` to `PATH`, reopen the terminal, and verify with `tesseract --version`.

### `ImportError: cannot import name 'recognize_image'`

Stop the old Streamlit process with `Ctrl+C`, make sure the terminal is in this project folder, and restart:

```powershell
cd "C:\Users\hp\Desktop\COMPUTERVISIONPP\robust-ocr-pipeline"
\.venv\Scripts\Activate.ps1
streamlit run app.py
```

### `File does not exist: app.py`

The command was run from `C:\Users\hp\Desktop\COMPUTERVISIONPP` instead of the project folder. Run `cd` to the project folder first or use the absolute Streamlit command shown above.

### OCR quality is poor

Use a sharp, well-lit image with large text. Avoid heavy blur, extreme skew, severe perspective distortion, very low-light images, complex backgrounds, and very small text. Compare the attempts in the browser before changing preprocessing parameters.

## Limitations

This is a lightweight classical OCR pipeline. It does not perform full perspective correction, deep-learning text detection, language-specific correction, or handwriting recognition. Results depend on image quality, font, layout, language data installed with Tesseract, and the selected Tesseract version.

## Future Improvements

- Perspective correction for photographed documents.
- Better illumination correction and deblurring.
- Language selection in the UI.
- OCR confidence visualization by word.
- CER/WER evaluation against labeled text.
- PaddleOCR comparison.
- Data augmentation and a larger evaluation dataset.
