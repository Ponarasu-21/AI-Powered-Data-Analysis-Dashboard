# AI-Powered Data Analysis Dashboard

This project is a Streamlit web application for data preprocessing, visualization, correlation analysis, and basic AI insights.

## Task Objective
Build a web-based dashboard that:
- Imports a CSV dataset
- Displays dataset statistics
- Handles missing values
- Generates visualizations
- Shows correlations between features
- Provides basic AI-generated insights

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- scikit-learn

## Files
- `app.py` - Streamlit dashboard implementation
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `.gitignore` - ignores environment and cache files

## Getting Started
1. Open the workspace in VS Code.
2. Create and activate a Python virtual environment.
   - Windows PowerShell:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```powershell
   streamlit run app.py
   ```

## Using the Dashboard
1. Upload a CSV dataset from the sidebar.
2. Review dataset statistics and preview the first rows.
3. Explore missing-value handling options.
4. View correlation heatmaps and visualizations.
5. Select a target column for basic AI/ML model evaluation.
