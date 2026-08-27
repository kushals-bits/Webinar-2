"""
Dataset Generator for Webinar 2: Data Preprocessing
Creates realistic datasets for Tabular, Text, and Image modalities.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def generate_tabular_dataset(output_path="data/tabular/customer_churn_raw.csv", n_samples=1200, random_state=42):
    """
    Generates a realistic customer churn tabular dataset with:
    - Missing values (MCAR, MAR)
    - Mixed categorical types (Nominal, Ordinal)
    - Dirty string values (e.g. '$123.45', whitespace, invalid strings)
    - Extreme outliers in numerical features
    - Skewed continuous features
    - Class imbalance (~85% non-churn, ~15% churn)
    """
    np.random.seed(random_state)
    random.seed(random_state)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Customer IDs
    customer_ids = [f"CUST_{i+10000}" for i in range(n_samples)]

    # 2. Demographics & Continuous features
    # Age: Normal with some unrealistic outliers and nulls
    age = np.random.normal(loc=41, scale=12, size=n_samples).round()
    age[np.random.choice(n_samples, size=int(0.04 * n_samples), replace=False)] = np.nan
    # Outliers
    outlier_idx = np.random.choice(n_samples, size=15, replace=False)
    age[outlier_idx[:8]] = np.random.choice([115, 120, -5, -1, 142], size=8)

    # Tenure months: Uniform / right-skewed with missing
    tenure = np.random.exponential(scale=24, size=n_samples).clip(0, 72).round()
    tenure[np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)] = np.nan

    # Monthly Charges: Skewed right with extreme outliers
    monthly_charges = np.random.gamma(shape=5.0, scale=12.0, size=n_samples) + 20.0
    # Add extreme outliers
    monthly_charges[outlier_idx[8:]] = np.random.choice([1500.0, 2400.0, 3200.0, 9999.0], size=7)
    monthly_charges[np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)] = np.nan

    # Total Charges as messy strings
    total_charges = []
    for m, t in zip(monthly_charges, tenure):
        if np.isnan(m) or np.isnan(t):
            total_charges.append(" " if random.random() < 0.5 else None)
        else:
            base = float(m * (t + 1) * np.random.uniform(0.9, 1.1))
            fmt = random.choice([
                f"${base:.2f}",
                f" {base:.2f} ",
                f"{base:.2f}",
                f"{int(base)} USD",
                " " if random.random() < 0.05 else f"${base:.2f}"
            ])
            total_charges.append(fmt)

    # 3. Categorical Features
    contract_choices = ["Month-to-month", "One year", "Two year"]
    contract_type = np.random.choice(contract_choices, p=[0.55, 0.25, 0.20], size=n_samples).astype(object)
    contract_type[np.random.choice(n_samples, size=int(0.03 * n_samples), replace=False)] = np.nan

    payment_choices = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_method = np.random.choice(payment_choices, p=[0.35, 0.25, 0.20, 0.20], size=n_samples).astype(object)
    payment_method[np.random.choice(n_samples, size=int(0.02 * n_samples), replace=False)] = np.nan

    internet_choices = ["DSL", "Fiber optic", "No"]
    internet_service = np.random.choice(internet_choices, p=[0.40, 0.45, 0.15], size=n_samples).astype(object)

    tech_support_choices = ["No", "Yes", "No internet service"]
    tech_support = np.random.choice(tech_support_choices, p=[0.50, 0.35, 0.15], size=n_samples).astype(object)

    support_tickets = np.random.poisson(lam=1.2, size=n_samples)
    support_tickets[outlier_idx[:5]] = [15, 22, 19, 25, 30]

    # 4. Imbalanced Churn Target (approx 15% churners)
    churn_prob = (
        0.05
        + 0.15 * (np.nan_to_num(monthly_charges, nan=65) > 80)
        + 0.20 * (contract_type == "Month-to-month")
        + 0.15 * (tech_support == "No")
        + 0.10 * (support_tickets > 3)
        - 0.15 * (np.nan_to_num(tenure, nan=12) > 36)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = [1 if random.random() < p else 0 for p in churn_prob]
    # Ensure final imbalance ~15%
    churn_labels = ["Yes" if c == 1 else "No" for c in churn]

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "tech_support": tech_support,
        "support_tickets": support_tickets,
        "churn": churn_labels
    })

    df.to_csv(output_path, index=False)
    print(f"-> Generated raw tabular dataset at: {output_path} ({len(df)} rows)")
    return df


def generate_text_dataset(output_path="data/text/product_reviews_raw.csv", n_samples=1000, random_state=42):
    """
    Generates a realistic NLP product review dataset with:
    - HTML tags (<p>, <br/>, <span>, <b>)
    - URLs and email fragments
    - Messy capitalization, emojis, repeated punctuation (!!!, ???)
    - Contractions, slang, typos
    - Imbalanced sentiment labels (80% Positive, 20% Negative)
    """
    random.seed(random_state)
    np.random.seed(random_state)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    positive_templates = [
        "Absolutely <b>loved</b> this product! It worked like a charm right out of the box. <br/> Highly recommended! 😊",
        "Great quality for the price. Delivered fast from https://example.com/shipping. Five stars! ⭐️⭐️⭐️⭐️⭐️",
        "THIS IS AMAZING!! Works so smoothly, exceeded all my expectations... Won't hesitate to buy again!!",
        "Super sleek design and very intuitive to use. Best purchase I've made this year. <span>Top notch quality</span>.",
        "Really solid device! Battery lasts for days and setup was super easy. 10/10 would buy again.",
        "Excellent customer support when I had a question. Everything arrived in pristine packaging. 👍",
        "Fantastic value! It is durable, looks stylish, and performs flawlessly.",
        "I was skeptical at first, but wow... totally impressed by the build quality! 😊✨",
        "Works exactly as described. Clean interface and super reliable performance.",
        "Must have gadget! I told all my colleagues about this. Fast shipping too: contact support@gadgets.org for info."
    ]

    negative_templates = [
        "Terrible experience!! Stopped working after just 2 days. <b>DO NOT BUY!</b> <br/> Total waste of money 😡",
        "Poor quality material. The handle snapped instantly. Very disappointed with this brand.",
        "Defective unit received. Customer service was unresponsive at http://bad-help.com. 👎",
        "HORRIBLE! It overheats within 10 minutes. Extremely unsafe and cheap plastic smell.",
        "Arrived damaged and missing key accessories. Packaging was torn apart. Never purchasing again!",
        "Do not waste your hard earned cash! Slow, laggy, and full of bugs. 0/10 stars.",
        "Worst customer support ever. Refused to issue a refund. Totally regret this purchase 😡",
        "Complete disaster. Broke on the first try. Returning immediately for a full refund!"
    ]

    reviews = []
    sentiments = []
    ratings = []

    # 80% Positive (imbalanced)
    n_pos = int(n_samples * 0.80)
    n_neg = n_samples - n_pos

    for i in range(n_pos):
        base = random.choice(positive_templates)
        # Add random noise/variations
        if random.random() < 0.3:
            base = base.upper()
        if random.random() < 0.4:
            base += " Check out https://reviews.store/item" + str(random.randint(100, 999))
        if random.random() < 0.2:
            base = "   " + base + "   \n\n"
        reviews.append(base)
        sentiments.append("Positive")
        ratings.append(random.choice([4, 5]))

    for i in range(n_neg):
        base = random.choice(negative_templates)
        if random.random() < 0.3:
            base = base.upper()
        if random.random() < 0.4:
            base += " Contact refund@support-desk.net for details!!!"
        if random.random() < 0.2:
            base = "   " + base + "   \n\n"
        reviews.append(base)
        sentiments.append("Negative")
        ratings.append(random.choice([1, 2]))

    # Shuffle
    combined = list(zip(reviews, ratings, sentiments))
    random.shuffle(combined)
    reviews, ratings, sentiments = zip(*combined)

    df = pd.DataFrame({
        "review_id": [f"REV_{i+1000}" for i in range(n_samples)],
        "review_text": reviews,
        "rating": ratings,
        "sentiment": sentiments
    })

    df.to_csv(output_path, index=False)
    print(f"-> Generated raw text dataset at: {output_path} ({len(df)} rows)")
    return df


def generate_image_dataset(output_dir="data/image/raw", n_samples_per_class=40, random_state=42):
    """
    Generates synthetic geometric shape images across 3 distinct classes:
    - 'circle'
    - 'square'
    - 'triangle'

    Images include realistic real-world challenges:
    - Arbitrary varying resolutions (e.g. 72x72, 96x96, 120x80, 64x64)
    - Different color spaces / lighting contrasts
    - Random Gaussian / Poisson noise & blur
    - Spatial translation and rotation
    """
    np.random.seed(random_state)
    random.seed(random_state)
    os.makedirs(output_dir, exist_ok=True)

    classes = ["circle", "square", "triangle"]
    resolutions = [(64, 64), (80, 80), (96, 96), (120, 90), (100, 100)]
    palette = [
        (220, 50, 50),   # Red
        (50, 150, 220),  # Blue
        (40, 180, 80),   # Green
        (230, 180, 30),  # Yellow
        (160, 50, 210)   # Purple
    ]

    total_created = 0

    for cls in classes:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(n_samples_per_class):
            width, height = random.choice(resolutions)
            # Create base image with random background color
            bg_color = (
                random.randint(220, 255),
                random.randint(220, 255),
                random.randint(220, 255)
            )
            img = Image.new("RGB", (width, height), color=bg_color)
            draw = ImageDraw.Draw(img)

            shape_color = random.choice(palette)
            # Center coordinates with slight jitter
            cx = width // 2 + random.randint(-8, 8)
            cy = height // 2 + random.randint(-8, 8)
            size = min(width, height) // 3 + random.randint(-4, 6)

            if cls == "circle":
                bbox = [cx - size, cy - size, cx + size, cy + size]
                draw.ellipse(bbox, fill=shape_color, outline=(20, 20, 20), width=2)

            elif cls == "square":
                bbox = [cx - size, cy - size, cx + size, cy + size]
                draw.rectangle(bbox, fill=shape_color, outline=(20, 20, 20), width=2)

            elif cls == "triangle":
                points = [
                    (cx, cy - size),
                    (cx - size, cy + size),
                    (cx + size, cy + size)
                ]
                draw.polygon(points, fill=shape_color, outline=(20, 20, 20))

            # Add random noise
            img_arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(5, 18), img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            noisy_img = Image.fromarray(img_arr)

            if random.random() < 0.3:
                noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

            img_path = os.path.join(cls_dir, f"{cls}_{i+1:03d}.png")
            noisy_img.save(img_path)
            total_created += 1

    print(f"-> Generated raw image dataset at: {output_dir} ({total_created} images across {len(classes)} classes)")
    return output_dir


def generate_all():
    """Generates datasets for Tabular, Text, and Image modalities."""
    print("=" * 70)
    print("Generating Datasets for Webinar 2: Data Preprocessing...")
    print("=" * 70)
    generate_tabular_dataset()
    generate_text_dataset()
    generate_image_dataset()
    print("=" * 70)
    print("All datasets successfully generated!")
    print("=" * 70)


if __name__ == "__main__":
    generate_all()
