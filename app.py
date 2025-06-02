
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_analysis.py

A template script for:
 1. Fetching/loading a CSV dataset (local or remote URL)
 2. Inspecting/cleaning the data
 3. Performing basic exploratory data analysis
 4. Plotting distributions and correlations with matplotlib

Usage:
    python data_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURATION: 
#    - Modify these variables to point to your actual data source.
#    - DATA_SOURCE can be a local file path or a remote CSV URL.

# Example 1: Loading from a local CSV file:
#DATA_SOURCE = "path/to/your/local_dataset.csv"

# Example 2: Loading from a remote URL:
DATA_SOURCE = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
# (This is just an example CSV; replace with your own URL or path.)

# Specify an output folder for figures (created if it doesn't exist)
OUTPUT_FIG_DIR = "figures"
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


def fetch_data(source):
    """
    Fetch/load the dataset into a pandas DataFrame.
    Supports local file paths or HTTP(s) URLs.
    """
    try:
        if source.lower().startswith(("http://", "https://")):
            df = pd.read_csv(source)  # pandas can read from URL directly
        else:
            df = pd.read_csv(source)
    except Exception as e:
        print(f"Error loading data from {source}: {e}")
        sys.exit(1)
    return df


def basic_inspection(df):
    """
    Print basic information: shape, column dtypes, head, missing values.
    """
    print("\n=== Dataframe Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print("=== Column Data Types ===")
    print(df.dtypes, "\n")

    print("=== Head (first 5 rows) ===")
    print(df.head(), "\n")

    print("=== Summary of Missing Values ===")
    missing = df.isna().sum()
    print(missing[missing > 0] if any(missing > 0) else "No missing values detected.\n")


def clean_and_preprocess(df):
    """
    A placeholder function to demonstrate basic cleaning:
      - Drop duplicates
      - Impute numeric columns' missing values with median
      - Convert object‐dtype columns to categorical if needed
    """
    # 1) Drop exact-duplicate rows
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"Dropped {before - after} duplicate rows.\n")

    # 2) Identify numeric vs. non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # 3) Impute numeric columns' missing values with the median
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Imputed numeric column '{col}' missing values with median = {median_val}")

    # 4) (Optional) Convert object columns to category:
    for col in object_cols:
        num_unique = df[col].nunique(dropna=True)
        if num_unique < df.shape[0] * 0.5:
            # If cardinality is not too high, convert to category
            df[col] = df[col].astype("category")
            print(f"Converted '{col}' to 'category' dtype (unique values: {num_unique})")

    print("\n=== After Cleaning: Missing values check ===")
    missing_after = df.isna().sum()
    print(missing_after[missing_after > 0] if any(missing_after > 0) else "No missing values remain.\n")

    return df


def exploratory_data_analysis(df):
    """
    Perform EDA: 
      - Summary statistics for numeric columns
      - Value counts for categorical columns
      - Correlation matrix for numeric features
    """
    print("\n=== Descriptive Statistics (Numeric Columns) ===")
    print(df.describe().transpose(), "\n")

    print("=== Value Counts (Categorical Columns) ===")
    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            print(f"\n-- Column: {col} --")
            print(df[col].value_counts().head(10))
    else:
        print("No categorical columns detected.\n")

    # Correlation matrix (numeric columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        print("\n=== Correlation Matrix (Numeric Columns) ===")
        print(corr, "\n")
    else:
        print("Not enough numeric columns to compute correlations.\n")


def plot_histograms(df, numeric_cols):
    """
    Plot histograms for each numeric column.
    """
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.hist(df[col], bins=30, edgecolor="black", alpha=0.7)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_FIG_DIR, f"{col}_histogram.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved histogram: {output_path}")


def plot_boxplots(df, numeric_cols):
    """
    Plot boxplots for each numeric column to visualize outliers.
    """
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna(), vert=True)
        plt.title(f"Boxplot of {col}")
        plt.ylabel(col)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_FIG_DIR, f"{col}_boxplot.png")
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved boxplot: {output_path}")


def plot_correlation_heatmap(df, numeric_cols):
    """
    Plot a correlation heatmap (using matplotlib imshow).
    """
    if len(numeric_cols) < 2:
        print("Skipping correlation heatmap: not enough numeric columns.")
        return

    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(ticks=range(len(numeric_cols)), labels=numeric_cols, rotation=45, ha="right")
    plt.yticks(ticks=range(len(numeric_cols)), labels=numeric_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_FIG_DIR, "correlation_heatmap.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap: {output_path}")


def main():
    # Step 1: Fetch/load the data
    print("Loading data from:", DATA_SOURCE)
    df = fetch_data(DATA_SOURCE)

    # Step 2: Basic inspection
    basic_inspection(df)

    # Step 3: Cleaning & preprocessing
    df_clean = clean_and_preprocess(df)

    # Step 4: Exploratory Data Analysis
    exploratory_data_analysis(df_clean)

    # Step 5: Visualizations
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        print("\nGenerating histograms for numeric columns...")
        plot_histograms(df_clean, numeric_columns)

        print("\nGenerating boxplots for numeric columns...")
        plot_boxplots(df_clean, numeric_columns)

        print("\nGenerating correlation heatmap...")
        plot_correlation_heatmap(df_clean, numeric_columns)
    else:
        print("No numeric columns found—skipping plotting.")

    print("\nData analysis complete. Figures saved in:", OUTPUT_FIG_DIR)


if __name__ == "__main__":
    main()
