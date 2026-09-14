# Robust OCR Pipeline

A Python OCR project that extracts readable text from images using OpenCV and Tesseract OCR.

## Sample Input / Output

Sample input:

```text
sample_input/sample.png
```

Expected output:

```text
PathPal OCR Demo
Obstacle ahead - Distance: 12 meters
Crosswalk detected
```

OCR output can vary slightly depending on the image quality and Tesseract version.

## Approach

The pipeline compares multiple OCR candidates instead of forcing every image through one thresholding method:

1. Load and validate the image.
2. Upscale small images when appropriate.
3. Correct meaningful image skew.
4. Create original, minimal, CLAHE, Otsu, and adaptive-threshold candidates.
5. Run Tesseract with PSM 3, 6, and 11.
6. Compare word confidence, word count, and line count.
7. Select and clean the strongest result.

## Technologies

- Python
- OpenCV
- Tesseract OCR
- pytesseract
- Streamlit

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Tesseract on Windows:

```powershell
winget install --id UB-Mannheim.TesseractOCR -e
```

Verify the installation:

```powershell
tesseract --version
```

## Run Locally

Run the command-line OCR pipeline:

```powershell
python ocr_pipeline.py --image sample_input\sample.png
```

Run the Streamlit browser interface:

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal, usually `http://localhost:8501`.

## Deployed Demo

[Open the live Streamlit app](https://ocr-pipeline-computervision.streamlit.app/)
