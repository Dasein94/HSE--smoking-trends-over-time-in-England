# Health Survey for England (HSE): Smoking Behaviour Analysis (1999 vs 2019)

## Overview
This project analyses changes in smoking behaviour in England between 1999 and 2019 using HSE datasets (study number 4365 (1999 wave, 10259 x 1699), and the study number 8860 (2019 wave, 10299 x 1841)).  

The **goal** is to explore trends in cigarette smoking status (smokers versus non - smokers), and cigarette quantity smoked in 1999 and 2019, taking into consideration sample weights.

This work was completed as part of a group data analysis project, using colab. However, what it is shared here is my individual contribution in the project.

---

## Data

⚠️ Due to confidentiality restrictions, raw datasets are not included in this repository.

---

## Research Questions

- Has smoking prevalence changed between 1999 and 2019?
- Has the number of cigarettes smoked per day by adults changed over time?
- Has the frequency of cigarettes smoked by young people changed over time?

---

## Structure

```
Case study/
    ├── README.md
    ├── case report final.pdf
    ├── analyses.py
    ├── requirements.txt
    ├── .gitignore
    ├── figures/
        ├── df19 bars.png
        ├── df19 matrix.png
        ├── df99 bars.png
        ├── df99 matrix.png
        ├── smk boxplot final.png
        ├── smk prev final.png
        └── young ppl smk final.png
    └── Papers/
        ├── Saygin_2021.pdf
        └── systematic review.pdf
```

### File descriptions
- **case report final.pdf** – Report 
- **analyses.py** – Script
- **requirements.txt** – Recreate the environment - packages installed
- **figures/** – Contains graphs to visualize nulls (msno package) and results
- **Papers/** – Background information

## Methods

The analyses were conducted in Python using:

- Data cleaning and preprocessing with `pandas` and `numpy`
- Descriptive statistics and visualisation (`matplotlib`, `seaborn`, `missingno`)
- Inferential statistics:
  - Mann–Whitney U tests using `pingouin`
  - Logistic Regression model using `statsmodels`

Survey weighting was applied to ensure population representativeness.

---

## Key Findings

- Smoking prevalence was significantly lower in 2019 than in 1999 (β = -0.35, SE = 0.05, z = -7.86, p < .001). 
- The odds of smoking were 30% lower in 2019 than in 1999 (OR = 0.70, 95% CI [0.64, 0.77]) 
- There are no statistically significant differences in cigarette smoking frequency or quantity between the two years (p>.05) in adults and young smokers.

## Reproducibility

To recreate the environment:

```bash
pip install -r requirements.txt
