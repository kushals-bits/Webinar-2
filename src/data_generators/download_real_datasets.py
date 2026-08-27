"""
Real Practical Dataset Downloader for Webinar 2: Data Preprocessing.
Downloads and formats real-world practical datasets for Tabular, Text, and Image modalities.
"""

import os
import io
import requests
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_digits, fetch_20newsgroups


# ─────────────────────────────────────────────────────────────────────────────
# 1. TABULAR: IBM Telco Customer Churn (Real-world enterprise dataset - 7,043 rows)
# ─────────────────────────────────────────────────────────────────────────────
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def download_tabular_dataset(output_path="data/tabular/telco_churn_raw.csv") -> pd.DataFrame:
    """
    Downloads the real IBM Telco Customer Churn dataset (7,043 rows, 21 columns).
    Features: customer demographics, contract type, payment method, monthly/total charges, churn.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Downloading IBM Telco Customer Churn dataset (7,043 rows)...")
    try:
        resp = requests.get(TELCO_URL, timeout=15)
        if resp.status_code == 200:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            df = pd.read_csv(output_path)
            print(f"-> Tabular dataset ready: {df.shape[0]} rows x {df.shape[1]} cols at {output_path}")
            return df
        else:
            raise RuntimeError(f"HTTP Status {resp.status_code}")
    except Exception as e:
        print(f"Warning: Remote download failed ({e}). Loading fallback real-world dataset.")
        # Fallback to sklearn breast cancer or synthetic with real schema
        from sklearn.datasets import load_breast_cancer
        bc = load_breast_cancer(as_frame=True)
        df = bc.frame
        df.to_csv(output_path, index=False)
        return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT: 20 Newsgroups (Real internet newsgroup forum discussions)
# ─────────────────────────────────────────────────────────────────────────────
def download_text_dataset(output_path="data/text/newsgroups_raw.csv") -> pd.DataFrame:
    """
    Loads real internet discussion posts from 20 Newsgroups (sci.med vs alt.atheism).
    Contains realistic noisy text: email artifacts, quotations, jargon, abbreviations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading 20 Newsgroups dataset (sci.med vs alt.atheism)...")
    try:
        categories = ["sci.med", "alt.atheism"]
        news = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers"), random_state=42)
        
        # Filter non-empty texts
        valid_records = []
        for i, (text, target) in enumerate(zip(news.data, news.target)):
            clean_t = text.strip()
            if len(clean_t) > 20:  # Minimum substantive text length
                valid_records.append({
                    "doc_id": f"DOC_{i+1:04d}",
                    "raw_text": clean_t,
                    "category": categories[target],
                    "target": target
                })
        
        df = pd.DataFrame(valid_records)
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"-> Text dataset ready: {len(df)} documents at {output_path}")
        return df
    except Exception as e:
        print(f"Warning: 20 Newsgroups load error ({e}). Loading SMS real dataset fallback...")
        sms_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        resp = requests.get(sms_url, timeout=15)
        df_sms = pd.read_csv(io.StringIO(resp.text), sep="\t", names=["category", "raw_text"])
        df_sms["doc_id"] = [f"DOC_{i+1:04d}" for i in range(len(df_sms))]
        df_sms["target"] = df_sms["category"].map({"ham": 0, "spam": 1})
        df_sms.to_csv(output_path, index=False, encoding="utf-8")
        print(f"-> Text dataset ready: {len(df_sms)} SMS records at {output_path}")
        return df_sms


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE: Handwritten Digits (Real human handwriting - Digits 0, 1, 2)
# ─────────────────────────────────────────────────────────────────────────────
def download_image_dataset(output_dir="data/image/raw") -> str:
    """
    Loads real handwritten digit image dataset (classes: 0, 1, 2).
    Generates standardized PNG images from real human handwriting samples.
    """
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    print("Loading real handwritten digits dataset (classes 0, 1, 2)...")

    digits = load_digits()
    classes = [0, 1, 2]
    total_created = 0

    for cls in classes:
        cls_dir = os.path.join(output_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        idx = np.where(digits.target == cls)[0]

        for i, sample_idx in enumerate(idx):
            # Scale 8x8 pixels to 0-255 uint8
            raw_pixels = digits.images[sample_idx]
            scaled = (raw_pixels / 16.0 * 255.0).clip(0, 255).astype(np.uint8)
            # Resize to 28x28
            img = Image.fromarray(scaled, mode="L").resize((28, 28), Image.Resampling.NEAREST).convert("RGB")
            img.save(os.path.join(cls_dir, f"digit_{cls}_{i+1:03d}.png"))
            total_created += 1

    print(f"-> Image dataset ready: {total_created} real handwritten digit images at {output_dir}")
    return output_dir


def download_all():
    """Downloads and formats all real practical datasets."""
    print("=" * 80)
    print("Preparing Real Practical Datasets for Webinar 2...")
    print("=" * 80)
    download_tabular_dataset()
    download_text_dataset()
    download_image_dataset()
    print("=" * 80)
    print("All real practical datasets successfully prepared!")
    print("=" * 80)


if __name__ == "__main__":
    download_all()
