#!/usr/bin/env python3
"""
Task 1 – Image Metadata Analysis
- Extract EXIF metadata (camera make/model, datetime, orientation)
- Parse GPS (and convert to decimal lat/lon if present)

- Optional OCR of visible text and language detection (--ocr)
- Writes a human-readable report to stdout and (optionally) a file
"""

import argparse
import json
import os
import sys

from PIL import Image, ExifTags, ImageOps

# Optional deps for bonus
try:
    import pytesseract  # requires Tesseract OCR installed on system
except Exception:
    pytesseract = None

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

# Build reverse maps for EXIF tags
EXIF_TAGS = ExifTags.TAGS
GPS_TAGS = ExifTags.GPSTAGS

def _rational_to_float(x):
    """Convert PIL rationals/tuples to float safely."""
    try:
        if hasattr(x, "numerator") and hasattr(x, "denominator"):
            return float(x.numerator) / float(x.denominator)
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            return float(num) / float(den) if den else 0.0
        return float(x)
    except Exception:
        return None

def _dms_to_decimal(dms, ref):
    """Convert degrees/minutes/seconds tuple to signed decimal degrees."""
    if not dms or len(dms) != 3:
        return None
    d = _rational_to_float(dms[0]) or 0.0
    m = _rational_to_float(dms[1]) or 0.0
    s = _rational_to_float(dms[2]) or 0.0
    dec = d + (m / 60.0) + (s / 3600.0)
    if ref in ("S", "W"):
        dec = -dec
    return dec

def extract_exif(image: Image.Image):
    """Return a dict of EXIF with friendly keys + parsed GPS."""
    exif_raw = image.getexif()
    if not exif_raw:
        return {"exif": {}, "gps": {}}

    # Map tag IDs to names
    exif_named = {}
    for tag_id, value in exif_raw.items():
        name = EXIF_TAGS.get(tag_id, str(tag_id))
        exif_named[name] = value

    gps_info = exif_named.get("GPSInfo")
    gps_parsed = {}
    if isinstance(gps_info, dict):
        # Map GPS tag ids to names
        gps_named = {GPS_TAGS.get(k, str(k)): v for k, v in gps_info.items()}
        lat = lon = None
        lat_ref = gps_named.get("GPSLatitudeRef")
        lon_ref = gps_named.get("GPSLongitudeRef")
        lat_raw = gps_named.get("GPSLatitude")
        lon_raw = gps_named.get("GPSLongitude")
        if lat_raw and lat_ref:
            lat = _dms_to_decimal(lat_raw, lat_ref)
        if lon_raw and lon_ref:
            lon = _dms_to_decimal(lon_raw, lon_ref)

        gps_parsed = {
            "raw": {k: str(v) for k, v in gps_named.items()},
            "latitude_decimal": lat,
            "longitude_decimal": lon,
        }

    return {
        "exif": exif_named,
        "gps": gps_parsed,
    }

def try_ocr(image: Image.Image):
    """Run OCR and language detection if deps exist; return dict."""
    result = {"text": None, "language": None, "note": None}
    if pytesseract is None:
        result["note"] = "pytesseract not installed; skip OCR."
        return result
    # If Tesseract binary missing, pytesseract will raise — catch it
    try:
        # Respect EXIF orientation so text isn’t sideways
        img = ImageOps.exif_transpose(image)
        text = pytesseract.image_to_string(img)
        text = text.strip()
        result["text"] = text if text else None
        if text and detect is not None:
            try:
                result["language"] = detect(text)
            except Exception:
                result["language"] = None
        elif text and detect is None:
            result["note"] = "langdetect not installed; language not detected."
        return result
    except Exception as e:
        result["note"] = f"OCR failed: {e}"
        return result

def analyze_image(path, do_ocr=False):
    with Image.open(path) as img:
        info = {
            "file": {
                "name": os.path.basename(path),
                "path": os.path.abspath(path),
                "format": img.format,
                "mode": img.mode,
                "size": {"width": img.width, "height": img.height},
            }
        }
        exif_data = extract_exif(img)
        info.update(exif_data)

        # Common EXIF fields (may be absent)
        exif = exif_data.get("exif", {})
        info["summary"] = {
            "camera_make": exif.get("Make"),
            "camera_model": exif.get("Model"),
            "datetime_original": exif.get("DateTimeOriginal") or exif.get("DateTime"),
            "has_gps": bool(exif_data.get("gps", {}).get("latitude_decimal") is not None and
                            exif_data.get("gps", {}).get("longitude_decimal") is not None),
        }

        if do_ocr:
            info["ocr"] = try_ocr(img)

        return info

def write_report(data, out_path=None):
    # Pretty JSON to stdout
    print(json.dumps(data, indent=2, ensure_ascii=False))
    # Also write a human-readable report if requested
    if out_path:
        lines = []
        f = data["file"]
        s = data["summary"]
        lines.append(f"# Image Metadata Report: {f['name']}")
        lines.append("")
        lines.append("## File")
        lines.append(f"- Path: {f['path']}")
        lines.append(f"- Format: {f['format']}")
        lines.append(f"- Mode: {f['mode']}")
        lines.append(f"- Size: {f['size']['width']}x{f['size']['height']}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Camera: {s.get('camera_make')} {s.get('camera_model')}".strip())
        lines.append(f"- Date/Time: {s.get('datetime_original')}")
        lines.append(f"- GPS present: {s.get('has_gps')}")
        gps = data.get("gps", {})
        if gps:
            lines.append(f"- GPS (decimal): {gps.get('latitude_decimal')}, {gps.get('longitude_decimal')}")
        if "ocr" in data:
            o = data["ocr"]
            lines.append("")
            lines.append("## OCR")
            lines.append(f"- Language: {o.get('language')}")
            lines.append(f"- Note: {o.get('note')}")
            if o.get("text"):
                lines.append("")
                lines.append("### Extracted Text")
                lines.append(o["text"])
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser(description="Image metadata analysis (EXIF + optional OCR).")
    parser.add_argument("image", help="Path to image (e.g., sample.jpg)")
    parser.add_argument("--ocr", action="store_true", help="Run OCR and attempt language detection (bonus)")
    parser.add_argument("--out", default="output.txt", help="Write a readable report to this file (default: output.txt)")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    data = analyze_image(args.image, do_ocr=args.ocr)
    write_report(data, out_path=args.out)

if __name__ == "__main__":
    main()
