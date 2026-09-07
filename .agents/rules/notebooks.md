# Jupyter Notebook Execution Guidelines

Whenever Jupyter notebooks (`.ipynb`) in this repository are executed or modified:
1. **Attach Output After Execution**: Ensure all execution outputs (stdout, stderr, evaluation metrics, charts, tables, return values, display data) are attached to each code cell and saved in-place in the `.ipynb` file.
2. **Execute In-Place**: When running notebooks programmatically (via `ExecutePreprocessor`, `nbconvert`, or Jupyter APIs), persist the updated notebook structure with complete cell output blocks.
3. **Keep Results Current**: Never strip or discard cell outputs when updating or running notebooks, unless explicitly requested.
