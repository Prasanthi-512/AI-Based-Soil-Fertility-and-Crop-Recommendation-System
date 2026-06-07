import pdfplumber
import pandas as pd
import re

DEFAULTS = {
    "ph": 7.0,
    "n": 50,
    "p": 30,
    "k": 150,
    "oc": 0.8,
    "ec": 0.5,
    "moisture": 15
}

def try_float(val, field):
    try:
        if val is None:
            return DEFAULTS[field]
        return float(val)
    except:
        return DEFAULTS[field]

def parse_csv_file(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    row = df.iloc[0].to_dict()

    return {
        "ph": try_float(row.get("ph"), "ph"),
        "n": try_float(row.get("n"), "n"),
        "p": try_float(row.get("p"), "p"),
        "k": try_float(row.get("k"), "k"),
        "oc": try_float(row.get("oc"), "oc"),
        "ec": try_float(row.get("ec"), "ec"),
        "moisture": try_float(row.get("moisture"), "moisture")
    }

def parse_pdf_file(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"

    def find(field, patterns):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return try_float(m.group(1), field)
        return DEFAULTS[field]

    return {
        "ph": find("ph", [r"pH[:\s]*([0-9.]+)"]),
        "n": find("n", [r"N[:\s]*([0-9.]+)", r"Nitrogen[:\s]*([0-9.]+)"]),
        "p": find("p", [r"P[:\s]*([0-9.]+)", r"Phosphorus[:\s]*([0-9.]+)"]),
        "k": find("k", [r"K[:\s]*([0-9.]+)", r"Potassium[:\s]*([0-9.]+)"]),
        "oc": find("oc", [r"OC[:\s]*([0-9.]+)", r"Organic Carbon[:\s]*([0-9.]+)"]),
        "ec": find("ec", [r"EC[:\s]*([0-9.]+)"]),
        "moisture": find("moisture", [r"Moisture[:\s]*([0-9.]+)"]),
    }
