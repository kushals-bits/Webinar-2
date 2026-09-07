"""
Generates a beginner-friendly Microsoft Word (.docx) document specifically designed
to teach first-time Machine Learning students the core concepts of Data Preprocessing
across Tabular, Text, and Image modalities using simple analogies, clear steps, and visuals.
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


def add_beginner_callout(doc, title, text, emoji="💡", border_hex="2563EB", bg_hex="EFF6FF"):
    """Adds a friendly, highlighted callout box for beginner students."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=130, bottom=130, left=180, right=160)
    
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
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(f"{emoji} {title}\n")
    r_title.bold = True
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    r_text = p.add_run(text)
    r_text.font.name = "Calibri"
    r_text.font.size = Pt(10)
    r_text.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def format_table(table, col_widths, col_alignments=None):
    """Applies clean, beginner-readable table styling."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    for row_idx, row in enumerate(table.rows):
        is_header = (row_idx == 0)
        bg_color = "1E3A8A" if is_header else ("F8FAFC" if row_idx % 2 == 1 else "FFFFFF")
        
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            
            tcPr = cell._tc.get_or_add_tcPr()
            border_col = "3B82F6" if is_header else "E2E8F0"
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


def build_beginner_docx(output_path="BEGINNER_DATA_PREPROCESSING_GUIDE.docx"):
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
    run_badge = title_p.add_run("BEGINNER MACHINE LEARNING SERIES\n")
    run_badge.font.name = "Calibri"
    run_badge.font.size = Pt(10.5)
    run_badge.bold = True
    run_badge.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
    
    run_title = title_p.add_run("A Beginner's Guide to Data Preprocessing")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.bold = True
    run_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_after = Pt(14)
    run_sub = subtitle_p.add_run(
        "A friendly, step-by-step introduction to preparing Tabular (Numbers), Text (Words), and Image (Pixels) data "
        "for Machine Learning models, explained with simple real-world examples and everyday analogies."
    )
    run_sub.font.size = Pt(11)
    run_sub.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: WHAT IS DATA PREPROCESSING?
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 1: 🌟 What is Data Preprocessing? (The Cooking Analogy)")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    doc.add_paragraph(
        "Welcome to Machine Learning! Before an algorithm can learn from data, we must prepare the data so the computer can understand it."
    )

    add_beginner_callout(
        doc,
        "The Master Chef Analogy: Why Preprocessing is Essential",
        "Imagine you want to cook a delicious gourmet dinner. You buy fresh vegetables from the market. "
        "Can you throw unwashed, unpeeled potatoes with soil and stems straight into a hot pan? Of course not! "
        "You must wash the mud off (Cleaning), cut them into uniform bite-sized pieces (Formatting & Resizing), "
        "and balance the ingredients so salt doesn't overpower everything (Scaling & Rebalancing).\n\n"
        "In Machine Learning, your Algorithm is the Oven, and your Data is the Raw Food. "
        "If you feed dirty data into the model, you get bad results. In data science, we call this:\n"
        "👉 'Garbage In, Garbage Out' (GIGO).",
        emoji="🍳", border_hex="2563EB", bg_hex="EFF6FF"
    )

    doc.add_paragraph(
        "In this course, we explore the three main types of data you will encounter in real-world jobs:"
    )

    data_types = [
        ("1. 📊 Tabular Data (Spreadsheets): ", "Rows and columns of customer records, prices, ages, and categories (like Microsoft Excel)."),
        ("2. 💬 Text Data (Natural Language): ", "Unstructured sentences, customer reviews, tweets, forum messages, and emails."),
        ("3. 🖼️ Image Data (Computer Vision): ", "Photographs, medical scans, and handwritten digits made of tiny colored dots called pixels.")
    ]
    for b_txt, n_txt in data_types:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r_b = p.add_run(b_txt)
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.add_run(n_txt)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: TABULAR PREPROCESSING (SPREADSHEETS)
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 2: 📊 Tabular Data Preprocessing (Spreadsheets & Records)")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    doc.add_paragraph(
        "In our project, we work with a real dataset from IBM containing 7,043 telecom customers. "
        "Our goal is to predict which customers are about to cancel their service (called 'Churn'). Here are the 5 basic steps to prepare tabular data:"
    )

    # Step 1: Missing values
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 1: Handling Missing Values & Dirty Strings")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph(
        "• The Problem: Sometimes people leave blanks in a form, or new customers with 0 months have blank spaces (' ') for their Total Spend.\n"
        "• Why not just delete those rows? If you delete every row with a missing value, you throw away valuable data!\n"
        "• The Solution (Imputation): We fill missing numbers with the Median (middle number) and missing words with the Mode (most common category)."
    )

    # Step 2: Categorical Encoding
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 2: Turning Words into Numbers (Categorical Encoding)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph(
        "Computers cannot do math on words like 'Electronic Check' or 'Credit Card'. We must convert categories into numbers using two methods:"
    )

    # Table of encoding
    enc_tbl = doc.add_table(rows=3, cols=4)
    enc_widths = [Inches(1.5), Inches(1.5), Inches(1.5), Inches(2.0)]
    
    enc_headers = ["Encoding Type", "When to Use", "Simple Example", "How it Works"]
    for i, h in enumerate(enc_headers):
        enc_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    enc_rows = [
        ("One-Hot Encoding\n(Nominal)", "Categories with NO ranking or order", "Colors: Red, Blue, Green\nPayment: Mail, Card", "Creates separate 0/1 columns for each option (e.g. Is_Card = 1, Is_Mail = 0)."),
        ("Ordinal Encoding\n(Ranked)", "Categories that have a clear natural order", "T-Shirt Sizes:\nSmall < Medium < Large", "Assigns ranked numbers:\nSmall = 0, Medium = 1, Large = 2.")
    ]
    for row_idx, data in enumerate(enc_rows, start=1):
        for col_idx, text in enumerate(data):
            enc_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(enc_tbl, enc_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Step 3: Scaling
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 3: Feature Scaling (Putting Numbers on an Equal Footing)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    add_beginner_callout(
        doc,
        "Why Scaling Matters: The Age vs. Salary Problem",
        "Imagine your model looks at two numbers: Age (e.g., 30 years) and Annual Salary (e.g., $75,000). "
        "Because 75,000 is mathematically so much larger than 30, the model thinks Salary is 2,500 times more important than Age! "
        "Scaling shrinks all numbers into a fair, balanced range so no single column dominates.",
        emoji="⚖️", border_hex="059669", bg_hex="ECFDF5"
    )

    doc.add_paragraph(
        "• StandardScaler (Z-Score): Centers numbers around 0 with spread 1. Great for normal data, but gets pulled by extreme outliers.\n"
        "• MinMaxScaler: Squeezes everything into the range [0.0, 1.0].\n"
        "• RobustScaler (Used in our repo): Uses the Median and the middle 50% (IQR). If a billionaire enters the dataset, RobustScaler ignores the extreme number and scales regular customer data properly!"
    )

    if os.path.exists("reports/tabular_scaling_and_outliers.png"):
        doc.add_picture("reports/tabular_scaling_and_outliers.png", width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 1: Notice how RobustScaler (bottom right) keeps the distribution clean even with outliers!")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # Step 4: Imbalance & SMOTE
    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 4: Class Imbalance & SMOTE (The Lazy Student Analogy)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    add_beginner_callout(
        doc,
        "Class Imbalance: The Lazy Student Analogy",
        "Imagine a multiple-choice test with 100 questions where 95 answers are 'A' and only 5 answers are 'B'. "
        "A lazy student who didn't study at all can just guess 'A' for every single question and score a 95% grade! "
        "Did the student actually learn anything? No! They failed to identify any of the important 'B' answers.\n\n"
        "In our churn dataset, ~74% of customers stay and only ~26% leave. If a model guesses 'No Churn' every time, "
        "it looks 74% accurate, but fails to catch any leaving customers.\n"
        "👉 SMOTE solves this by creating smart, synthetic examples of leaving customers so the model learns both sides equally.",
        emoji="🎯", border_hex="D97706", bg_hex="FFFBEB"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: TEXT PREPROCESSING (NLP)
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 3: 💬 Text (NLP) Preprocessing (Words & Sentences)")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    doc.add_paragraph(
        "Natural Language Processing (NLP) is how computers read human language. "
        "In our project, we classify 1,780 internet forum messages between Medical discussions (sci.med) and Philosophy discussions (alt.atheism)."
    )

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 1: Cleaning Messy Text in 7 Simple Stages")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    stages = [
        ("1. Strip HTML Tags: ", "Remove website code like '<p>' or '<br>' and turn '&amp;' into '&'."),
        ("2. Remove URLs & Emails: ", "Delete 'https://...' and 'john@email.com' because web addresses don't describe topics."),
        ("3. Expand Contractions: ", "Turn 'won't' into 'will not' and 'can't' into 'cannot' so the negative meaning is clear."),
        ("4. Lowercasing: ", "Make all letters lowercase so 'Doctor' and 'doctor' are treated as the exact same word."),
        ("5. Remove Punctuation: ", "Delete symbols like '!@#$%^&*()' so words like 'heart!' match 'heart'."),
        ("6. Remove Stopwords: ", "Delete boring common words like 'the', 'is', 'at', 'which', and 'on' that appear in every sentence."),
        ("7. Clean Whitespace: ", "Collapse multiple spaces and line breaks into clean, spaced words.")
    ]
    for b_txt, n_txt in stages:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r_b = p.add_run(b_txt)
        r_b.bold = True
        p.add_run(n_txt)

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 2: TF-IDF & N-Grams (Giving Important Words Higher Scores)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph(
        "• What is TF-IDF? TF-IDF stands for Term Frequency - Inverse Document Frequency. "
        "It gives a high score to unique words that describe a specific topic (like 'chemotherapy' or 'theology') "
        "and a low score to generic words that appear everywhere.\n"
        "• Sublinear Scaling (1 + log(tf)): If a word appears 10 times in an article, is it really 10 times more important than if it appeared once? "
        "No! Logarithmic scaling prevents repetitive words from dominating the score.\n"
        "• Bigrams (Two-Word Pairs): Single words (unigrams) miss context. For example, 'happy' vs 'not happy'. "
        "By looking at two words together (bigrams), the computer understands phrases like 'clinical trial' or 'immune system'."
    )

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 3: Linguistic Style Features (How People Write)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph(
        "Besides vocabulary, the writing style gives strong clues! We extract 11 statistical features:\n"
        "• Shouting Ratio: How much text is in ALL CAPS?\n"
        "• Punctuation Density: How many question marks ('?') or exclamation points ('!') are used?\n"
        "• Lexical Diversity: Does the author use a rich variety of different words or repeat the same 5 words?\n"
        "• Sentiment Score: Are there more positive words ('great', 'cured') or negative words ('pain', 'broken')?"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # PART 4: IMAGE PREPROCESSING (COMPUTER VISION)
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 4: 🖼️ Image Preprocessing (Pixels & Shapes)")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    doc.add_paragraph(
        "In Computer Vision, our dataset consists of 537 real human handwriting images of the digits 0, 1, and 2. "
        "To a computer, an image is just a grid of numbers where 0 represents Black and 255 represents White."
    )

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 1: Resizing & Letterboxing (No More Squished Photos!)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    add_beginner_callout(
        doc,
        "Letterboxing: The Widescreen Movie Analogy",
        "Have you ever watched a wide cinema movie on a standard television? If the TV forces the movie to fill the screen, "
        "people look unnaturally stretched and tall! Instead, the TV adds black bars on the top and bottom (padding) "
        "so the movie keeps its original shape.\n\n"
        "Letterboxing does the exact same thing for images: it scales the digit proportionally and pastes it onto a centered canvas "
        "so the curve of a '0' or line of a '1' is never squished or distorted.",
        emoji="📺", border_hex="7C3AED", bg_hex="F5F3FF"
    )

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 2: Normalization (Dividing by 255)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    doc.add_paragraph(
        "Raw pixel values are integers from 0 to 255. Dividing each pixel by 255.0 converts them into smooth decimals between 0.0 and 1.0. "
        "Small numbers prevent math errors and help algorithms train much faster."
    )

    h2 = doc.add_heading(level=2)
    h2.paragraph_format.space_before = Pt(12)
    h2.paragraph_format.space_after = Pt(4)
    r2 = h2.add_run("Step 3: PCA Compression (The 3D Shadow Analogy)")
    r2.font.color.rgb = RGBColor(0x1E, 0x40, 0xAF)

    add_beginner_callout(
        doc,
        "PCA Compression: The Shadow Puppet Analogy",
        "A 28x28 image has 2,352 individual pixel numbers. Most background pixels are completely empty, and neighboring pixels are nearly identical.\n\n"
        "Imagine holding your hand up to a light to make a shadow puppet. Your hand is a complex 3D object with millions of cells, "
        "but the 2D shadow on the wall captures the entire shape of a bird or dog with just an outline!\n\n"
        "PCA (Principal Component Analysis) compresses 2,352 raw pixel numbers down to just 21 essential summary numbers (99.1% compression!) "
        "while keeping 95% of all visual information.",
        emoji="👤", border_hex="0284C7", bg_hex="F0F9FF"
    )

    if os.path.exists("reports/image_pca_and_reconstruction.png"):
        doc.add_picture("reports/image_pca_and_reconstruction.png", width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 2: Notice how the reconstructed digit on the right looks nearly identical to the original, built from just 21 numbers!")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5: BEFORE VS AFTER BENCHMARKS
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 5: 📈 The Proof: Before vs. After Results")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    doc.add_paragraph(
        "Here is what happens when we compare naive models trained on raw data against models trained on preprocessed data:"
    )

    bench_tbl = doc.add_table(rows=6, cols=5)
    bench_widths = [Inches(1.2), Inches(1.5), Inches(1.1), Inches(1.1), Inches(1.6)]
    
    b_headers = ["Modality", "Metric", "Before (Raw)", "After (Cleaned)", "Why it Matters for Beginners"]
    for i, h in enumerate(b_headers):
        bench_tbl.cell(0, i).paragraphs[0].add_run(h)
        
    b_rows = [
        ("📊 Tabular (Churn)", "Churner Recall", "52.89%", "61.46% (+16.2%)", "SMOTE & XGBoost catch 16.2% more customers before they leave."),
        ("📊 Tabular (Churn)", "Minority F1", "0.5888", "0.5998", "Balanced detection of churning customers."),
        ("💬 Text (Forum)", "Accuracy", "90.34%", "95.96% (+5.6%)", "Cleaning & TF-IDF jump accuracy to ~96%."),
        ("💬 Text (Forum)", "ROC-AUC", "0.9037", "0.9877 (+9.3%)", "Model confidence ranking is near-perfect."),
        ("🖼️ Image (Digits)", "Dimensions", "2,352 dims", "21 dims (-99.1%)", "99.1% smaller size with 99.26% accuracy.")
    ]
    for row_idx, data in enumerate(b_rows, start=1):
        for col_idx, text in enumerate(data):
            bench_tbl.cell(row_idx, col_idx).paragraphs[0].add_run(text)
            
    format_table(bench_tbl, bench_widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    if os.path.exists("reports/before_vs_after_benchmarks.png"):
        doc.add_picture("reports/before_vs_after_benchmarks.png", width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = cap.add_run("Figure 3: Visual comparison showing Green (Preprocessed) beating Red (Baseline) across all metrics!")
        r_cap.font.size = Pt(8.5)
        r_cap.italic = True
        r_cap.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    # ─────────────────────────────────────────────────────────────────────────
    # PART 6: BEGINNER GLOSSARY
    # ─────────────────────────────────────────────────────────────────────────
    h1 = doc.add_heading(level=1)
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(6)
    r = h1.add_run("Part 6: 📖 Beginner Glossary: 10 Essential Terms")
    r.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)

    glossary_terms = [
        ("1. Imputation: ", "Filling in missing blanks in a dataset with sensible guesses like the median or most common category."),
        ("2. One-Hot Encoding: ", "Converting unranked words into separate 0/1 binary columns so models can do math on them."),
        ("3. Ordinal Encoding: ", "Converting ranked categories into ordered numbers (e.g. Small=0, Medium=1, Large=2)."),
        ("4. Feature Scaling: ", "Adjusting all numbers into a fair, balanced range so large numbers don't overpower small numbers."),
        ("5. Outlier: ", "An extreme or impossible value (e.g. a customer age of 250) that skews normal statistical calculations."),
        ("6. Class Imbalance: ", "When one category appears far more often than another (e.g. 95% stay vs 5% leave)."),
        ("7. SMOTE: ", "A smart algorithm that creates realistic synthetic examples of rare events to balance the dataset."),
        ("8. Stopwords: ", "Common grammar words ('the', 'is', 'at') that are removed in NLP because they contain no topic information."),
        ("9. TF-IDF: ", "A mathematical formula that scores words higher if they are unique and informative to a specific document."),
        ("10. PCA: ", "Principal Component Analysis — a compression technique that reduces thousands of pixels into a few summary features.")
    ]

    for b_txt, n_txt in glossary_terms:
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        r_b = p.add_run(b_txt)
        r_b.bold = True
        r_b.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        p.add_run(n_txt)

    # Save document
    doc.save(output_path)
    print(f"Successfully generated Beginner Word document at: {output_path}")


if __name__ == "__main__":
    build_beginner_docx()
