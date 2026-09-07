"""
Generates a comprehensive, publication-grade Microsoft Word (.docx) document
specifically detailing all theoretical, mathematical, and algorithmic concepts
used across the Data Preprocessing Masterclass repository.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_background(cell, fill_hex):
    """Sets cell background fill color."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    """Sets inner cell padding."""
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


def add_callout(doc, title, text, border_hex="4F46E5", bg_hex="F8FAFC"):
    """Adds a callout box with a left accent border."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
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


def add_formula_block(doc, formula_title, formula_text, explanation_text):
    """Adds a stylized mathematical formula block."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=120, bottom=120, left=180, right=180)
    
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:left w:val="single" w:sz="18" w:space="0" w:color="0284C7"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'<w:right w:val="single" w:sz="6" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2)
    
    r_head = p.add_run(f"📐 Mathematical Formulation: {formula_title}\n")
    r_head.bold = True
    r_head.font.name = "Calibri"
    r_head.font.size = Pt(10.5)
    r_head.font.color.rgb = RGBColor(0x03, 0x69, 0xA1)
    
    r_form = p.add_run(f"{formula_text}\n")
    r_form.bold = True
    r_form.font.name = "Consolas"
    r_form.font.size = Pt(10)
    r_form.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
    
    r_exp = p.add_run(f"Concept: {explanation_text}")
    r_exp.font.name = "Calibri"
    r_exp.font.size = Pt(9.5)
    r_exp.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def format_table(table, col_widths, col_alignments=None):
    """Applies professional table formatting."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    for row_idx, row in enumerate(table.rows):
        is_header = (row_idx == 0)
        bg_color = "1E1B4B" if is_header else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
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


def build_concepts_docx(output_path="CORE_CONCEPTS_AND_THEORY_GUIDE.docx"):
    doc = docx.Document()
    
    # 0.75 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # ─────────────────────────────────────────────────────────────────────────
    # COVER / HEADER
    # ─────────────────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(10)
    title_p.paragraph_format.space_after = Pt(4)
    run_badge = title_p.add_run("THEORETICAL & MATHEMATICAL FOUNDATIONS\n")
    run_badge.font.name = "Calibri"
    run_badge.font.size = Pt(10)
    run_badge.bold = True
    run_badge.font.color.rgb = RGBColor(0x4F, 0x46, 0xE5)
    
    run_title = title_p.add_run("Data Preprocessing: Core Concepts Guide")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(14)
    run_sub = subtitle_p.add_run(
        "An in-depth curriculum explaining the statistical theory, mathematical formulations, "
        "algorithmic trade-offs, and empirical benchmarks across Tabular, Text (NLP), and Image (CV) modalities."
    )
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: THE BEFORE VS AFTER PARADIGM
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("1. 🌟 The 'Before vs. After' Paradigm & The Empirical Delta (Δ)")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "A common misconception in data science education is that selecting a sophisticated algorithm (e.g., XGBoost, "
        "Deep Neural Networks) can compensate for dirty or unscaled input data. In reality, modern ML models are strictly "
        "garbage-in, garbage-out systems."
    )

    add_formula_block(
        doc,
        "The Empirical Delta Framework",
        "Delta = Score(Preprocessed Pipeline) - Score(Baseline Pipeline)\n"
        "Relative Improvement (%) = (Delta / Score(Baseline)) * 100",
        "Quantifies the exact mathematical lift contributed purely by data cleaning, scaling, rebalancing, and compression."
    )

    doc.add_paragraph(
        "By enforcing this dual-pipeline architecture across all three modalities, students learn to attribute performance gains "
        "specifically to data preprocessing rather than random hyperparameter variance."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: TABULAR PREPROCESSING CONCEPTS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("2. 📊 Tabular Preprocessing Concepts & Formulations")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Tabular data (rows = observations, columns = features) is the bedrock of enterprise data science. "
        "The IBM Telco Customer Churn dataset represents customer profiles, contracts, and retention statuses."
    )

    # Outlier Detection
    add_formula_block(
        doc,
        "Interquartile Range (IQR) vs. Z-Score Outlier Detection",
        "IQR = Q3 - Q1\n"
        "Lower Bound = Q1 - 1.5 * IQR,   Upper Bound = Q3 + 1.5 * IQR\n"
        "Z-Score = |(x - μ) / σ| > 3.0",
        "Z-score assumes Gaussian normality and is vulnerable to outlier distortion because extreme values inflate μ and σ. "
        "IQR relies on non-parametric order statistics (25th and 75th percentiles), making it immune to extreme skewness."
    )

    # Categorical Encoding
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("2.1 Nominal vs. Ordinal Categorical Encoding")
    r2.font.color.rgb = RGBColor(0x31, 0x2E, 0x81)

    doc.add_paragraph(
        "Categorical variables cannot be fed directly into mathematical loss functions. Two distinct strategies must be applied:"
    )

    enc_tbl = doc.add_table(rows=3, cols=4)
    enc_widths = [Inches(1.3), Inches(1.5), Inches(1.8), Inches(1.9)]
    
    enc_headers = ["Encoding Method", "Mathematical Nature", "Appropriate Data Type", "Dataset Example"]
    for i, h in enumerate(enc_headers):
        enc_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    enc_rows = [
        ("One-Hot Encoding\n(drop='first')", "Binary indicator columns\n[0, 1] per category", "Nominal variables with NO intrinsic hierarchy or order", "PaymentMethod (Electronic check, Mailed check, Bank transfer)"),
        ("Ordinal Encoding", "Ranked integers\n(0, 1, 2, ...)", "Ordinal variables with natural, monotonic ordering", "Contract (Month-to-month = 0 < One year = 1 < Two year = 2)")
    ]
    for row_idx, data in enumerate(enc_rows, start=1):
        for col_idx, text in enumerate(data):
            enc_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(enc_tbl, enc_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Scalers
    add_formula_block(
        doc,
        "Feature Scaling Formulations",
        "StandardScaler:  z = (x - μ) / σ\n"
        "MinMaxScaler:    x_scaled = (x - x_min) / (x_max - x_min)\n"
        "RobustScaler:    x_scaled = (x - Median) / IQR",
        "StandardScaler centers on mean and scales by variance. MinMaxScaler bounds values to [0, 1]. "
        "RobustScaler subtracts the median and divides by IQR, guaranteeing that extreme spending outliers do not distort typical customer values."
    )

    if os.path.exists("reports/tabular_scaling_and_outliers.png"):
        doc.add_picture("reports/tabular_scaling_and_outliers.png", width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 1: Empirical Distribution Comparison Across Scaling Strategies under Outliers")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # SMOTE
    add_formula_block(
        doc,
        "SMOTE (Synthetic Minority Over-sampling Technique)",
        "x_new = x_i + λ * (x_zi - x_i),    where λ ~ Uniform(0, 1)",
        "Rather than simply duplicating minority class rows (which leads to severe tree overfitting), SMOTE finds "
        "k-nearest neighbors in Euclidean feature space and interpolates new synthetic samples along the connecting vectors."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: TEXT PREPROCESSING CONCEPTS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("3. 💬 Text / NLP Preprocessing Concepts & Formulations")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Natural language text is unstructured and high-dimensional. Raw forum posts from 20 Newsgroups (sci.med vs alt.atheism) "
        "contain noise, typos, HTML tags, email artifacts, and slang."
    )

    # 7-Stage Cleaning
    doc.add_paragraph(
        "1. HTML Stripping: Removes markup tags and unescapes entities (&amp; -> &).\n"
        "2. URL & Email Stripping: Regex eliminates web addresses and author signatures.\n"
        "3. Contraction Expansion: Normalizes colloquialisms ('won't' -> 'will not', 'can't' -> 'cannot').\n"
        "4. Case Normalization: Lowercases text so 'Treatment' and 'treatment' map to identical vector tokens.\n"
        "5. Punctuation Removal: Strips noise characters while preserving alphanumeric tokens.\n"
        "6. Stopword Pruning: Filters high-frequency, non-discriminative grammar words ('the', 'is', 'at', 'which').\n"
        "7. Whitespace Collapsing: Standardizes multi-space and newline formatting into clean token streams."
    )

    # TF-IDF
    add_formula_block(
        doc,
        "TF-IDF with Sublinear Scaling & Document Cutoffs",
        "TF-IDF(t, d, D) = TF_sublinear(t, d) * IDF(t, D)\n"
        "TF_sublinear(t, d) = 1 + log(TF(t, d))   (if TF > 0, else 0)\n"
        "IDF(t, D) = log((1 + |D|) / (1 + DF(t))) + 1",
        "Sublinear TF scaling prevents a term appearing 20 times from having 20x the weight of a term appearing once. "
        "IDF discounts words that appear in nearly every document, highlighting rare, topic-specific vocabulary."
    )

    # Linguistic Features
    doc.add_paragraph(
        "Numerical Feature Extraction captures stylistic and structural signals independent of vocabulary: "
        "character length, word count, uppercase shouting ratio, punctuation density, lexical diversity (Unique Words / Total Words), "
        "and sentiment polarity score."
    )

    # Platt Calibration
    add_formula_block(
        doc,
        "Platt Scaling for Support Vector Machine Probability Calibration",
        "P(y = 1 | x) = 1 / (1 + exp(A * f(x) + B))",
        "LinearSVC optimizes the maximum-margin hyperplane (hinge loss), outputting geometric distances f(x) rather than "
        "probabilities. Platt scaling fits a logistic sigmoid model over the decision margins via CalibratedClassifierCV, "
        "enabling proper posterior probability estimation for ROC-AUC evaluation."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: IMAGE PREPROCESSING CONCEPTS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("4. 🖼️ Image / Computer Vision Preprocessing Concepts")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Computer vision preprocessing converts raw pixel matrices into standardized, normalized, and compressed feature representations."
    )

    # Letterboxing
    doc.add_paragraph(
        "• Letterboxing (Aspect-Preserving Resizing): Direct resizing scales width and height independently, squishing and "
        "distorting the aspect ratio of handwritten characters. Letterboxing scales the image uniformly and pastes it onto a "
        "centered canvas with neutral background padding, preserving true stroke geometry."
    )

    # MinMax Normalization
    add_formula_block(
        doc,
        "Pixel Intensity Normalization",
        "x_norm = x / 255.0    =>    x_norm in [0.0, 1.0]",
        "Raw 8-bit integer pixels range from 0 to 255. Scaling to [0.0, 1.0] stabilizes gradient calculations, prevents numerical "
        "overflow, and accelerates convergence in distance-based and kernel algorithms."
    )

    # PCA Derivation
    add_formula_block(
        doc,
        "Principal Component Analysis (PCA) Dimensionality Reduction",
        "Covariance Matrix:  Σ = (1 / N) * X_centered^T * X_centered\n"
        "Eigen-decomposition: Σ * v_i = λ_i * v_i\n"
        "Projection:         X_reduced = X_centered * W_k\n"
        "Reconstruction:     X_reconstructed = X_reduced * W_k^T + μ",
        "PCA identifies orthogonal eigenvectors (principal components) that maximize explained variance. "
        "By selecting components retaining 95% cumulative variance, 2,352 raw pixels are compressed into just 21 dimensions (99.1% compression)."
    )

    if os.path.exists("reports/image_pca_and_reconstruction.png"):
        doc.add_picture("reports/image_pca_and_reconstruction.png", width=Inches(6.2))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 2: PCA Cumulative Explained Variance (Scree Plot) and Digit Reconstruction from 21 Dimensions")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: DOWNSTREAM EVALUATION & METRIC TRADE-OFFS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("5. 📈 Downstream Evaluation & Metric Trade-Offs")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    doc.add_paragraph(
        "Evaluating models on real-world imbalanced datasets requires looking beyond raw accuracy."
    )

    add_formula_block(
        doc,
        "Classification Performance Metrics",
        "Precision = TP / (TP + FP)       (Quality of positive alarms)\n"
        "Recall    = TP / (TP + FN)       (Ability to catch all positive events)\n"
        "F1-Score  = 2 * (Precision * Recall) / (Precision + Recall) (Harmonic Mean)\n"
        "ROC-AUC   = Area Under the Receiver Operating Characteristic Curve",
        "In imbalanced problem spaces (e.g. churn detection), a model predicting 'No Churn' for all accounts scores 74% accuracy "
        "but 0% Recall. Preprocessing rebalances the decision space, trading a small amount of precision for a massive gain in Recall."
    )

    bench_tbl = doc.add_table(rows=6, cols=6)
    bench_widths = [Inches(1.0), Inches(1.3), Inches(1.0), Inches(1.1), Inches(0.9), Inches(1.2)]
    
    b_headers = ["Modality", "Metric", "Baseline", "Preprocessed", "Delta", "Impact"]
    for i, h in enumerate(b_headers):
        bench_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    b_rows = [
        ("📊 Tabular", "Churner Recall", "52.89%", "61.46%", "+8.57%", "+16.2% more churners detected"),
        ("📊 Tabular", "Minority F1", "0.5888", "0.5998", "+0.0110", "Balanced precision & recall"),
        ("💬 Text", "Accuracy", "90.34%", "95.96%", "+5.62%", "~96% topic classification"),
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
        r_cap = cap.add_run("Figure 3: Empirical Performance Benchmark Comparison Across All Three Modalities")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: SUMMARY REFERENCE CHEAT-SHEET
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("6. 📋 Core Concepts Cheat-Sheet & Code Mapping")
    r.font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)

    cheat_tbl = doc.add_table(rows=12, cols=4)
    cheat_widths = [Inches(1.5), Inches(1.0), Inches(1.6), Inches(2.4)]
    
    c_headers = ["Preprocessing Technique", "Modality", "Code Module", "Mathematical / Practical Purpose"]
    for i, h in enumerate(c_headers):
        cheat_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    c_rows = [
        ("IQR Outlier Detection", "Tabular", "src/tabular/profiling.py", "Non-parametric quartile bounds immune to skewness"),
        ("RobustScaler", "Tabular", "src/tabular/scaling.py", "Median and IQR scaling; resistant to extreme spending outliers"),
        ("One-Hot Encoding", "Tabular", "src/tabular/encoding.py", "Binary indicator matrix with drop='first' for nominal features"),
        ("Ordinal Encoding", "Tabular", "src/tabular/encoding.py", "Monotonic integer mapping preserving natural rank hierarchy"),
        ("SMOTE Oversampling", "Tabular", "src/tabular/imbalance.py", "Feature-space vector interpolation for minority class rebalancing"),
        ("7-Stage Regex Cleaner", "Text", "src/text/cleaning.py", "Strips HTML, emails, punctuation, contractions, and stopwords"),
        ("Sublinear TF-IDF", "Text", "src/text/vectorization.py", "Logarithmic term frequency dampening 1 + log(tf) with bigrams"),
        ("Linguistic Metadata", "Text", "src/text/feature_engineering.py", "Quantifies writing style, shouting ratio, and sentiment polarity"),
        ("Platt Calibration", "Text", "src/text/imbalance.py", "Fits sigmoid over SVM margins for calibrated probability outputs"),
        ("Letterbox Resizing", "Image", "src/image/resizing.py", "Preserves aspect ratio via centered background padding"),
        ("MinMax Normalization", "Image", "src/image/normalization.py", "Scales [0, 255] pixels to [0.0, 1.0] for gradient stability"),
    ]
    for row_idx, data in enumerate(c_rows, start=1):
        for col_idx, text in enumerate(data):
            cheat_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(cheat_tbl, cheat_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # Save document
    doc.save(output_path)
    print(f"Successfully generated Concepts Word document at: {output_path}")


if __name__ == "__main__":
    build_concepts_docx()
