# Digital Platform Competition and Rental Outcomes

This repository contains the code for a research project presented at the Stanford SC² Conference examining how simulated digital-platform competition, multihoming, and market concentration relate to rental-market outcomes.

## Overview

Using the UCI Apartments for Rent dataset, this project builds a simulated platform environment across Craigslist, Zillow, and Apartments.com and analyzes relationships between platform structure and outcomes such as rental price, time-on-market, listing exposure, and rental probability.

The project combines econometric analysis with machine learning to study how digital-market structure may influence housing-market outcomes.

## Methods

The analysis includes:

- Data cleaning and preprocessing
- Simulated heterogeneous platform assignment
- Platform-level price dispersion
- Landlord multihoming and listing exposure
- Neighborhood-level Herfindahl-Hirschman Index (HHI)
- LightGBM regression and classification models
- SHAP feature-attribution analysis
- Partial-dependence analysis for platform concentration
- OLS robustness checks with robust standard errors

## Repository Structure

```text
digital-housing-platform-analysis/
├── README.md
├── requirements.txt
├── housing_model.py
└── ols_robustness_checks.ipynb
```

### `housing_model.py`

Main modeling pipeline containing:

- UCI dataset loading and cleaning
- Simulated platform assignment
- Construction of platform competition variables
- Feature engineering
- LightGBM model training
- Model evaluation
- SHAP analysis
- Platform-concentration partial dependence

### `ols_robustness_checks.ipynb`

Econometric robustness analysis using OLS models with robust standard errors.

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Snet101/digital-housing-platform-analysis.git
cd digital-housing-platform-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main modeling pipeline:

```bash
python housing_model.py
```

The robustness notebook can be opened with Jupyter Notebook or JupyterLab.

## Research Context

This project was developed as an exploratory study of digital-platform competition in rental markets and was presented at the Stanford SC² Conference.

The analysis examines concepts including:

- platform concentration
- multihoming
- listing visibility
- price dispersion
- rental success
- time-on-market

## Limitations

Several platform-level variables are simulated because the underlying rental dataset does not contain observed platform-assignment or multihoming behavior.

The results should therefore be interpreted as exploratory and illustrative rather than as causal estimates of real-world platform effects.

## Tech Stack

- Python
- pandas
- NumPy
- LightGBM
- SHAP
- scikit-learn
- matplotlib
- seaborn
- statsmodels
- Jupyter Notebook

## Authors

Sanvi Nethikunta  
Rithik Gumpu
