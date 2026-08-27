"""
Real Dataset Downloader for Webinar 2: Data Preprocessing.
Downloads or loads practical, publicly available datasets for Tabular, Text, and Image modalities.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import urllib.request


# ─────────────────────────────────────────────────────────────────────────────
# 1. TABULAR: IBM Telco Customer Churn (Real-world business dataset)
# ─────────────────────────────────────────────────────────────────────────────
TELCO_CHURN_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_tabular_dataset(output_path="data/tabular/telco_churn_raw.csv") -> pd.DataFrame:
    """
    Downloads the IBM Telco Customer Churn dataset (7,043 real customer records).
    Features: Demographics, services subscribed, account info, binary churn target.
    - Real missing values in TotalCharges (whitespace)
    - Mixed categorical (nominal + ordinal) columns
    - Natural class imbalance (~27% Churn)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Downloading IBM Telco Customer Churn dataset...")
    try:
        urllib.request.urlretrieve(TELCO_CHURN_URL, output_path)
        df = pd.read_csv(output_path)
        print(f"-> Downloaded Telco Churn: {df.shape[0]} rows × {df.shape[1]} cols → {output_path}")
        return df
    except Exception as e:
        print(f"Download failed ({e}). Falling back to sklearn synthetic generation.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT: 20 Newsgroups — Real internet forum posts (sci.med vs alt.atheism)
# ─────────────────────────────────────────────────────────────────────────────
def download_text_dataset(output_path="data/text/newsgroups_raw.csv") -> pd.DataFrame:
    """
    Loads the 20 Newsgroups dataset (real internet posts from the early 1990s):
    - Classes: 'sci.med' (medical forum) vs 'alt.atheism' (philosophy forum)
    - Real noise: email headers, quoted replies, raw whitespace, code snippets
    - Natural imbalance corrected by subset selection
    Uses sklearn's built-in fetcher (no API key required).
    """
    from sklearn.datasets import fetch_20newsgroups
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Loading 20 Newsgroups dataset (sci.med vs alt.atheism)...")
    
    # Fetch both classes
    categories = ["sci.med", "alt.atheism"]
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers"),  # Keep body text only (realistic NLP challenge)
        random_state=42
    )
    
    labels = ["sci.med" if t == 0 else "alt.atheism" for t in data.target]
    df = pd.DataFrame({
        "doc_id": [f"DOC_{i+1:04d}" for i in range(len(data.data))],
        "raw_text": data.data,
        "category": labels,
        "target": data.target
    })

    df.to_csv(output_path, index=False)
    print(f"-> Loaded Newsgroups: {len(df)} documents, {len(set(labels))} classes → {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. IMAGE: MNIST Handwritten Digits — Real handwritten digit images (0 vs 1 vs 2)
# ─────────────────────────────────────────────────────────────────────────────
def download_image_dataset(output_dir="data/image/raw") -> str:
    """
    Loads the MNIST handwritten digits dataset (real human handwriting):
    - Classes: Digits 0, 1, 2 (first 3 classes for a manageable multi-class problem)
    - Each image: 28×28 pixels, grayscale
    - Samples 150 images per class (450 total) for efficient experimentation
    Saves each digit class as PNG images in class-named subdirectories.
    Uses sklearn's digits dataset (no API key required).
    """
    from sklearn.datasets import fetch_openml
    from PIL import Image
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    print("Loading MNIST handwritten digits dataset (classes: 0, 1, 2)...")
    
    try:
        # Try full MNIST via OpenML
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data, mnist.target.astype(int)
    except Exception:
        # Fallback to sklearn digits dataset (8x8 pixels)
        from sklearn.datasets import load_digits
        digits = load_digits()
        X, y = digits.images.reshape(len(digits.images), -1), digits.target
        print("  (Using sklearn digits fallback — 8x8 pixel images)")

    classes = [0, 1, 2]
    n_per_class = 150
    total_created = 0

    for cls in classes:
        cls_dir = os.path.join(output_dir, str(cls))
        os.makedirs(cls_dir, exist_ok=True)
        idx = np.where(y == cls)[0][:n_per_class]
        
        for i, sample_idx in enumerate(idx):
            pixels = X[sample_idx]
            
            # Handle 784-dim (28x28) or 64-dim (8x8) MNIST variants
            if len(pixels) == 784:
                img_arr = pixels.reshape(28, 28).astype(np.uint8)
            else:
                img_arr = (pixels / pixels.max() * 255).reshape(8, 8).astype(np.uint8)
            
            img = Image.fromarray(img_arr, mode="L").convert("RGB")  # Grayscale → RGB
            img.save(os.path.join(cls_dir, f"digit_{cls}_{i+1:03d}.png"))
            total_created += 1

    print(f"-> Saved MNIST digit images: {total_created} images ({n_per_class}/class) → {output_dir}")
    return output_dir


def download_all():
    """Downloads all real practical datasets for Webinar 2."""
    print("=" * 70)
    print("Downloading Real Practical Datasets for Webinar 2...")
    print("=" * 70)

    # 1. Tabular
    df = download_tabular_dataset()
    if df is None:
        print("  Telco Churn download failed — will use synthetic fallback in pipeline.")

    # 2. Text
    download_text_dataset()

    # 3. Image
    download_image_dataset()

    print("=" * 70)
    print("All practical datasets ready!")
    print("=" * 70)


if __name__ == "__main__":
    download_all()
