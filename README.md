# Applied AI & ML Essentials – Capstone Project

## Part 1: Data Preparation and Exploratory Data Analysis (EDA)

## Project Overview

This project is the first part of the Applied AI & ML Essentials Capstone. The objective is to perform data preparation and exploratory data analysis (EDA) on the Medical Insurance Cost Dataset. The analysis focuses on understanding the dataset, identifying potential data quality issues, exploring relationships between variables, and generating visualizations to gain meaningful insights.

---

## Dataset

**Dataset Name:** Medical Insurance Cost Dataset

The dataset contains information about individuals and their medical insurance charges. It includes both numerical and categorical features, making it suitable for exploratory analysis and regression modeling.

### Features

| Feature  | Description                                         |
| -------- | --------------------------------------------------- |
| age      | Age of the individual                               |
| sex      | Gender of the individual                            |
| bmi      | Body Mass Index                                     |
| children | Number of dependent children                        |
| smoker   | Smoking status (Yes/No)                             |
| region   | Residential region                                  |
| charges  | Individual medical insurance cost (Target Variable) |

---

## Technologies Used

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Project Structure

```text
part1/
│
├── data/
│   └── insurance.csv
│
├── plots/
│   ├── heatmap.png
│   ├── histograms.png
│   ├── charges_boxplot.png
│   ├── bmi_boxplot.png
│   ├── gender_count.png
│   ├── smoker_count.png
│   ├── region_count.png
│   ├── bmi_vs_charges.png
│   ├── age_vs_charges.png
│   ├── pairplot.png
│   ├── smoker_vs_charges.png
│   ├── gender_vs_charges.png
│   └── region_vs_charges.png
│
├── part1.ipynb
├── requirements.txt
└── README.md
```

---

## Data Preparation

The following preprocessing steps were performed:

* Loaded the dataset using Pandas.
* Inspected the dataset structure using `head()`, `info()`, and `describe()`.
* Checked for missing values.
* Checked for duplicate records and removed them if present.
* Identified numerical and categorical features.
* Generated summary statistics for numerical columns.

---

## Exploratory Data Analysis

The following visualizations were created:

* Correlation Heatmap
* Histograms for numerical features
* Boxplots for BMI and Insurance Charges
* Count plots for Gender, Smoking Status, and Region
* Scatter plots:

  * BMI vs Charges
  * Age vs Charges
* Pairplot
* Boxplots comparing insurance charges by:

  * Smoking Status
  * Gender
  * Region

All figures are automatically saved in the `plots/` directory.

---

## Key Findings

* The dataset contains both numerical and categorical variables, making it suitable for regression analysis.
* No missing values were found in the dataset.
* Duplicate records were checked and handled where necessary.
* Medical insurance charges generally increase with age.
* Smokers tend to have significantly higher insurance charges than non-smokers.
* BMI shows a moderate positive relationship with insurance charges.
* Regional differences in insurance charges are relatively small compared to the impact of smoking status.

---

## How to Run

1. Clone this repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Open `part1.ipynb` using Jupyter Notebook or Visual Studio Code.
4. Run all cells sequentially.
5. The generated visualizations will be saved automatically in the `plots/` folder.

---

## Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

## Conclusion

This project successfully performs data preparation and exploratory data analysis on the Medical Insurance Cost Dataset. The insights obtained from the analysis provide a strong foundation for building predictive machine learning models in the next phase of the capstone project.
