"""Confidence-guided OCR using several lightweight OpenCV image variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import pytesseract


PSM_MODES = (3, 6, 11)


@dataclass
class OCRAttempt:
    """The text and quality signals produced by one OCR configuration."""

    method: str
    psm: int
    image: object
    text: str
    mean_confidence: float
    word_count: int
    line_count: int

    @property
    def score(self) -> float:
        """Combine confidence and useful text quantity into one ranking value."""
        return self.mean_confidence * 0.75 + min(self.word_count, 60) * 0.5 + min(self.line_count, 20)


def _upscale_if_small(image):
    """Upscale modest images once without creating unnecessarily huge inputs."""
    height, width = image.shape[:2]
    if max(height, width) < 2400:
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return image.copy()


def deskew_image(image):
    """Rotate only when dark text indicates a meaningful document skew."""
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, foreground = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(foreground)
    if coordinates is None or len(coordinates) < 20:
        return image
    angle = cv2.minAreaRect(coordinates)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, rotation, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_minimal(image):
    """Keep character shapes intact with grayscale and light denoising."""
    return cv2.GaussianBlur(_grayscale(image), (3, 3), 0)


def preprocess_clahe(image):
    """Improve local contrast without throwing away grayscale information."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(_grayscale(image))


def preprocess_otsu(image):
    """Use a global Otsu threshold for clean, evenly lit documents."""
    denoised = cv2.GaussianBlur(_grayscale(image), (3, 3), 0)
    _, thresholded = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded


def preprocess_adaptive(image):
    """Use local thresholding only as one candidate for uneven lighting."""
    denoised = cv2.GaussianBlur(_grayscale(image), (3, 3), 0)
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


def preprocess_variants(image) -> dict[str, object]:
    """Build the original-preserving image plus four controlled variants."""
    prepared = deskew_image(_upscale_if_small(image))
    return {
        "Original": prepared,
        "Minimal": preprocess_minimal(prepared),
        "CLAHE contrast": preprocess_clahe(prepared),
        "Otsu threshold": preprocess_otsu(prepared),
        "Adaptive threshold": preprocess_adaptive(prepared),
    }


def clean_text(text: str) -> str:
    """Remove empty lines and whitespace while preserving line structure."""
    lines = [
        " ".join(line.replace("—", "-").replace("–", "-").split())
        for line in text.splitlines()
    ]
    return "\n".join(line for line in lines if line).strip()


def calculate_confidence(data: dict) -> tuple[float, int, int]:
    """Calculate mean word confidence, word count, and non-empty line count."""
    confidences = []
    words = []
    line_keys = set()
    for index, raw_text in enumerate(data.get("text", [])):
        word = raw_text.strip()
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if word and confidence >= 0:
            words.append(word)
            confidences.append(confidence)
            line_keys.add((data.get("block_num", [0])[index], data.get("par_num", [0])[index], data.get("line_num", [0])[index]))
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return mean_confidence, len(words), len(line_keys)


def run_ocr(image, method: str, psm: int) -> OCRAttempt:
    """Run Tesseract data and text extraction for one candidate image."""
    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
    raw_text = pytesseract.image_to_string(image, config=config)
    mean_confidence, word_count, line_count = calculate_confidence(data)
    return OCRAttempt(method, psm, image, clean_text(raw_text), mean_confidence, word_count, line_count)


def select_best_result(attempts: list[OCRAttempt]) -> OCRAttempt:
    """Select the candidate with the strongest confidence and text signals."""
    return max(attempts, key=lambda attempt: (attempt.score, attempt.mean_confidence))


def recognize_image(image, psm_modes: Sequence[int] = PSM_MODES) -> tuple[OCRAttempt, list[OCRAttempt]]:
    """Compare every preprocessing/PSM combination and return the winner."""
    attempts = [run_ocr(candidate, method, psm) for method, candidate in preprocess_variants(image).items() for psm in psm_modes]
    return select_best_result(attempts), attempts


def preprocess_image(image):
    """Compatibility wrapper returning the minimal candidate."""
    return preprocess_variants(image)["Minimal"]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Extract text from an image with OCR.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image")
    parser.add_argument("--save-preprocessed", type=Path, help="Optional best image output path")
    parser.add_argument("--output", type=Path, help="Optional extracted text output path")
    parser.add_argument("--psm", type=int, choices=PSM_MODES, help="Run only one PSM; default compares all modes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run confidence-guided OCR and return a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.image.is_file():
        print(f"Error: image file not found: {args.image}")
        return 1
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"Error: could not read a valid image: {args.image}")
        return 1
    try:
        best, _ = recognize_image(image, (args.psm,) if args.psm else PSM_MODES)
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR is not installed or is not available on PATH.")
        return 1
    except OSError as error:
        print(f"Error: Tesseract OCR could not run: {error}")
        return 1
    if not best.text:
        print("Error: OCR completed, but no text was detected.")
        return 1
    print(best.text)
    print(f"\nSelected: {best.method} + PSM {best.psm} ({best.mean_confidence:.1f}% confidence)")
    if args.save_preprocessed:
        args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_preprocessed), best.image)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(best.text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
