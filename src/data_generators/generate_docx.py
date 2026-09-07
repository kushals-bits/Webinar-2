"""
Generates a publication-grade, professionally styled Microsoft Word (.docx) document
for the Data Preprocessing Masterclass Student Study Guide.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets the background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets inner padding for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def add_callout(doc, title, text, border_hex="4F46E5", bg_hex="F3F4F6"):
    """Adds a callout block with a left accent border and shaded background."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    # Custom left border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_hex}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    r_title = p.add_run(f"📌 {title}\n")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)
    
    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(0x37, 0x41, 0x51)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def add_code_block(doc, code_text):
    """Adds a shaded code snippet block with monospaced font."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "1E293B")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="334155"/>'
        f'<w:left w:val="single" w:sz="4" w:space="0" w:color="334155"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="334155"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="334155"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    
    r = p.add_run(code_text.strip())
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0xE2, 0xE8, 0xF0)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def format_table(table, col_widths, col_alignments=None):
    """Styles a standard data table with branded headers, alternating rows, and grid lines."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    for row_idx, row in enumerate(table.rows):
        is_header = (row_idx == 0)
        bg_color = "1E1B4B" if is_header else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
            # Cell border styling
            tcPr = cell._tc.get_or_add_tcPr()
            border_col = "4F46E5" if is_header else "E2E8F0"
            borders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{border_col}"/>'
                f'<w:left w:val="none"/>'
                f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{border_col}"/>'
                f'<w:right w:val="none"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(borders)
            
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                if col_alignments and col_idx < len(col_alignments):
                    paragraph.alignment = col_alignments[col_idx]
                
                for run in paragraph.runs:
                    run.font.name = "Calibri"
                    if is_header:
                        run.bold = True
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    else:
                        run.font.size = Pt(9.5)
                        run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)


def build_docx(output_path="STUDENT_GUIDE.docx"):
    doc = docx.Document()
    
    # Page setup - 0.75 in margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Style Defaults
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    # ─────────────────────────────────────────────────────────────────────────
    # DOCUMENT COVER / HEADER
    # ─────────────────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(12)
    title_p.paragraph_format.space_after = Pt(4)
    run_badge = title_p.add_run("WEBINAR 2 MASTERCLASS COMPANION\n")
    run_badge.font.name = "Calibri"
    run_badge.font.size = Pt(10)
    run_badge.bold = True
    run_badge.font.color.rgb = RGBColor(0x4F, 0x46, 0xE5)
    
    run_title = title_p.add_run("End-to-End Data Preprocessing Guide")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(16)
    run_sub = subtitle_p.add_run(
        "A Comprehensive Reference on Data Hygiene, Feature Engineering, Class Rebalancing, "
        "Dimensionality Reduction, and Machine Learning Benchmarks across Tabular, Text, and Image Modalities."
    )
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # Metadata table
    meta_tbl = doc.add_table(rows=2, cols=2)
    meta_widths = [Inches(3.25), Inches(3.25)]
    meta_tbl.cell(0, 0).paragraphs[0].add_run("🎯 Target Audience: ML / Data Science Students")
    meta_tbl.cell(0, 1).paragraphs[0].add_run("📦 Modalities: Tabular, Text (NLP), Image (CV)")
    meta_tbl.cell(1, 0).paragraphs[0].add_run("🔬 Datasets: IBM Telco Churn, 20 Newsgroups, MNIST")
    meta_tbl.cell(1, 1).paragraphs[0].add_run("⚡ Key Models: XGBoost, LinearSVC, SVM RBF, KNN")
    format_table(meta_tbl, meta_widths)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: THE BIG PICTURE
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("1. 🌟 The Big Picture: Why Preprocessing Matters")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "In enterprise machine learning, predictive failure rarely stems from choosing the wrong algorithm — "
        "it almost always stems from poor data hygiene. Real-world raw data suffers from four pervasive challenges:"
    )

    challenges = [
        ("Dirty & Incomplete Records: ", "Missing entries, blank whitespace strings (' '), negative ages, and conflicting currency symbols crash production pipelines."),
        ("Non-Numeric Representations: ", "Mathematical estimators require structured numeric tensors. Strings, nominal labels, and nested categories must be encoded without introducing artificial numeric bias."),
        ("Severe Class Imbalance: ", "When rare target events (e.g., customer churn, medical diagnosis, fraud) represent only 10–25% of the data, naive classifiers predict the majority class for everyone."),
        ("The Curse of Dimensionality: ", "High-dimensional spaces (e.g., thousands of raw pixels or massive text vocabularies) cause severe overfitting, collinearity, and quadratic training slowdowns.")
    ]
    for bold_text, normal_text in challenges:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r_b = p.add_run(bold_text)
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.add_run(normal_text)

    add_callout(
        doc,
        "The Core Educational Principle: Before vs. After (Δ)",
        "Every single pipeline track in this codebase directly evaluates a Baseline Pipeline (raw/unprocessed data "
        "on naive models) against a Preprocessed Pipeline (cleaned, scaled, rebalanced, engineered data on modern algorithms) "
        "so students can measure the exact mathematical benefit (Delta) of every preprocessing stage.",
        border_hex="4F46E5", bg_hex="EEF2FF"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: ARCHITECTURE & WORKFLOW
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("2. 🏗️ Repository Architecture & Workflow")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "The repository implements a unified three-column architecture that maps each data modality "
        "through specialized preprocessing stages into downstream evaluation:"
    )

    arch_tbl = doc.add_table(rows=4, cols=4)
    arch_widths = [Inches(1.2), Inches(1.8), Inches(1.7), Inches(1.8)]
    
    headers = ["Modality", "Real Dataset", "Baseline Pipeline", "Preprocessed Pipeline"]
    for i, h in enumerate(headers):
        arch_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    data_rows = [
        ("📊 Tabular", "IBM Telco Churn\n(7,043 customer accounts)", "Naive Logistic Regression\n(unscaled, un-imputed)", "XGBoost Classifier +\nRobustScaler + SMOTE"),
        ("💬 Text (NLP)", "20 Newsgroups\n(1,780 sci.med vs alt.atheism)", "CountVectorizer (BoW) +\nSGD Classifier", "LinearSVC (Calibrated) +\nTF-IDF (1,2) + Feat Eng"),
        ("🖼️ Image (CV)", "MNIST Handwritten Digits\n(537 images: 0, 1, 2)", "Logistic Regression\n(2,352 raw unscaled pixels)", "SVM (RBF) & KNN on\n21 PCA components")
    ]
    for row_idx, data in enumerate(data_rows, start=1):
        for col_idx, text in enumerate(data):
            arch_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(arch_tbl, arch_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: TABULAR PREPROCESSING DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("3. 📊 Track 1: Tabular Preprocessing Deep-Dive")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Located in src/tabular/, this track tackles structured enterprise churn data from IBM. "
        "The objective is to predict customer subscription cancellation (Churn = Yes/No)."
    )

    # 3.1 Profiling
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("3.1 Data Health Profiling & Outlier Diagnostics (profiling.py)")
    r2.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)

    doc.add_paragraph(
        "Before writing transformation code, profiling evaluates column types, missingness, distribution skewness, "
        "and compares non-parametric vs parametric outlier detection:"
    )

    doc.add_paragraph(
        "• Interquartile Range (IQR Method): Non-parametric; computes IQR = Q3 - Q1. Bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR].\n"
        "• Z-Score Method: Parametric; flags points with |(x - mean) / std| > 3.0. Fails when extreme outliers distort the mean and standard deviation."
    )

    # 3.2 Cleaning & Encoding
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("3.2 Cleaning & Categorical Encoding (encoding.py)")
    r2.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)

    doc.add_paragraph(
        "In the IBM dataset, TotalCharges contains whitespace strings (' ') for new users (0 months tenure). "
        "The TabularEncoder handles median imputation for numeric features, mode imputation for categorical features, "
        "One-Hot Encoding for nominal columns (drop='first'), and Ordinal Encoding for ranked contracts:"
    )

    add_code_block(doc, """# One-Hot vs Ordinal Encoding in TabularEncoder
nominal_cols = ['PaymentMethod', 'InternetService', 'TechSupport']
ordinal_cols = ['Contract']
ordinal_cats = {'Contract': ['Month-to-month', 'One year', 'Two year']}

encoder = TabularEncoder(nominal_cols=nominal_cols, ordinal_cols=ordinal_cols, ordinal_categories=ordinal_cats)
X_train_enc = encoder.fit_transform(X_train_clean)""")

    # 3.3 Scaling
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("3.3 Scaling: StandardScaler vs MinMaxScaler vs RobustScaler (scaling.py)")
    r2.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)

    doc.add_paragraph(
        "Because customer spend features have heavy-tailed distributions with extreme outliers, RobustScaler "
        "is selected because it scales using the Median and IQR: x_scaled = (x - Median) / IQR, making it immune to outlier distortion."
    )

    if os.path.exists("reports/tabular_scaling_and_outliers.png"):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        doc.add_picture("reports/tabular_scaling_and_outliers.png", width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 1: Comparison of Feature Scalers under Outlier Distributions")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # 3.4 SMOTE
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("3.4 SMOTE Class Rebalancing & XGBoost Training (imbalance.py, model_trainer.py)")
    r2.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)

    doc.add_paragraph(
        "Because churn represents only ~26.5% of samples, SMOTE creates synthetic samples along the line segments "
        "connecting k-nearest minority neighbors in feature space: x_new = x_i + lambda * (x_zi - x_i)."
    )

    add_callout(
        doc,
        "Tabular Benchmark Result",
        "SMOTE + RobustScaler + XGBoost boosts Churner Recall from 52.89% to 61.46% (+16.2% relative gain), "
        "allowing customer success teams to detect significantly more at-risk clients.",
        border_hex="10B981", bg_hex="ECFDF5"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: TEXT PREPROCESSING DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("4. 💬 Track 2: Text / NLP Preprocessing Deep-Dive")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Located in src/text/, this pipeline processes raw internet discussion posts from 20 Newsgroups (sci.med vs alt.atheism). "
        "It executes 4 modular NLP phases:"
    )

    doc.add_paragraph(
        "1. 7-Stage Regex Cleaning (cleaning.py): HTML stripping, email/URL removal, contraction expansion ('won't' -> 'will not'), lowercasing, punctuation stripping, stopword filtering, and whitespace normalization.\n"
        "2. Sublinear TF-IDF Vectorization (vectorization.py): Extracts unigrams and bigrams with sublinear TF scaling (1 + log(tf)) so high-frequency words do not dominate vector space.\n"
        "3. Numerical Linguistic Features (feature_engineering.py): Computes 11 statistical features including uppercase shouting ratio, punctuation density, lexical diversity, and sentiment lexicon polarity.\n"
        "4. Calibrated LinearSVC (imbalance.py): Linear Support Vector Machine calibrated with Platt Scaling (CalibratedClassifierCV) to output genuine probabilities for ROC-AUC evaluation."
    )

    add_code_block(doc, """# Combining Cleaned TF-IDF Vectors + Scaled Linguistic Features
tfidf = TFIDFProcessor(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(clean_texts)

num_features = extract_numerical_text_features(raw_texts)
X_num_scaled = RobustScaler().fit_transform(num_features)

X_combined = np.hstack([X_tfidf, X_num_scaled])
svc_calibrated = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3).fit(X_combined, y)""")

    add_callout(
        doc,
        "Text Track Benchmark Result",
        "Preprocessing increases classification Accuracy from 90.34% to 95.96% (+5.62%) and boosts "
        "ROC-AUC from 0.9037 to 0.9877 (+9.3% relative improvement).",
        border_hex="06B6D4", bg_hex="ECFEFF"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: IMAGE PREPROCESSING DEEP-DIVE
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("5. 🖼️ Track 3: Image / Computer Vision Preprocessing Deep-Dive")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Located in src/image/, this track processes real MNIST handwritten digit images (classes 0, 1, and 2)."
    )

    doc.add_paragraph(
        "• Letterboxing Resizing (resizing.py): Scales rectangular images uniformly and pastes them onto a centered canvas with padding, preventing geometric stroke distortion.\n"
        "• Min-Max Normalization (normalization.py): Scales pixel intensities from [0, 255] to [0.0, 1.0] for numerical gradient stability.\n"
        "• PCA Dimensionality Reduction (pca_reduction.py): Compresses 2,352 raw pixel dimensions down to just 21 principal components (99.1% compression) while retaining 95% cumulative variance."
    )

    if os.path.exists("reports/image_pca_and_reconstruction.png"):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        doc.add_picture("reports/image_pca_and_reconstruction.png", width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 2: PCA Cumulative Explained Variance (Scree Plot) & Image Reconstruction")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: BENCHMARKS & EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("6. 📈 Downstream Evaluation & Master Benchmarks")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "The master evaluation engine (src/evaluation/) combines all test predictions and computes empirical deltas:"
    )

    bench_tbl = doc.add_table(rows=6, cols=6)
    bench_widths = [Inches(1.0), Inches(1.3), Inches(1.0), Inches(1.1), Inches(0.9), Inches(1.2)]
    
    b_headers = ["Modality", "Metric", "Baseline", "Preprocessed", "Delta", "Impact"]
    for i, h in enumerate(b_headers):
        bench_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    b_rows = [
        ("📊 Tabular", "Churner Recall", "52.89%", "61.46%", "+8.57%", "+16.2% more churners detected"),
        ("📊 Tabular", "Minority F1", "0.5888", "0.5998", "+0.0110", "Balanced precision & recall"),
        ("💬 Text", "Accuracy", "90.34%", "95.96%", "+5.62%", "~96% topic accuracy"),
        ("💬 Text", "ROC-AUC", "0.9037", "0.9877", "+0.0840", "+9.3% ranking boost"),
        ("🖼️ Image", "Dimensions", "2,352 dims", "21 dims", "-2,331", "99.1% compression (99.3% Acc)")
    ]
    for row_idx, data in enumerate(b_rows, start=1):
        for col_idx, text in enumerate(data):
            bench_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(bench_tbl, bench_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    if os.path.exists("reports/before_vs_after_benchmarks.png"):
        doc.add_picture("reports/before_vs_after_benchmarks.png", width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 3: Side-by-Side Performance Comparison (Baseline vs Preprocessed)")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: EXECUTION GUIDE
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("7. ⚙️ Execution Guide & Interactive Notebooks")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Students can execute the entire pipeline with one command or explore individual interactive Jupyter notebooks:"
    )

    add_code_block(doc, """# 1. Install required packages
pip install -r requirements.txt

# 2. Run master end-to-end pipeline CLI
python main.py

# 3. Launch interactive Jupyter Notebooks
jupyter notebook
# Open notebooks/01_tabular_preprocessing.ipynb, 02_text_preprocessing.ipynb, etc.""")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8: FAQ & INTERVIEW QUESTIONS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("8. 🧠 Conceptual Interview Questions (FAQ)")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    faqs = [
        ("Q1: Why not always use StandardScaler?",
         "StandardScaler uses the mean (mu) and standard deviation (sigma), which are heavily distorted by extreme outliers. RobustScaler uses the Median and IQR, making it immune to outlier distortion."),
        ("Q2: What is the difference between One-Hot Encoding and Ordinal Encoding?",
         "One-Hot Encoding creates binary indicator columns for nominal variables with no rank (e.g., Payment Method). Ordinal Encoding assigns integer ranks for variables with a natural order (e.g., Month-to-month < One year < Two year)."),
        ("Q3: Why does SMOTE generate synthetic points rather than duplicating minority rows?",
         "Row duplication causes decision trees to memorize specific points, causing severe overfitting. SMOTE synthesizes new, plausible points along line segments connecting neighboring minority samples in feature space."),
        ("Q4: Why is sublinear term frequency (1 + log(tf)) used in TF-IDF?",
         "A term appearing 20 times is not 20x more informative than one appearing 5 times. Sublinear scaling dampens high-frequency terms to keep vector representations balanced."),
        ("Q5: Why do we calibrate LinearSVC with Platt Scaling?",
         "LinearSVC optimizes hinge loss and outputs geometric margin distances rather than probabilities. CalibratedClassifierCV fits a logistic sigmoid curve on decision margins, enabling ROC-AUC scoring."),
        ("Q6: How does PCA achieve 99.1% dimensionality reduction without destroying accuracy?",
         "Adjacent image pixels share high spatial covariance. PCA rotates the coordinate system to orthogonal axes ordered by variance, concentrating 95% of all information in the top 21 components.")
    ]

    for q, a in faqs:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(6)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(6)
        p_a.add_run(a)

    # Save
    doc.save(output_path)
    print(f"Successfully generated Word document at: {output_path}")


if __name__ == "__main__":
    build_docx()
