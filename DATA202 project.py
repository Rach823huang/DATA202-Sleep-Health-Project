import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# load dataset
data = pd.read_csv('/Users/rachel_huang/DATA 202/Sleep_health_and_lifestyle_dataset.csv')

# preview
print(data.head())

# check shape
print("Shape of dataset:", data.shape)

# check column names
print("\nColumns:")
print(data.columns)

# check data types and non-null counts
print("\nDataset info:")
data.info()

# check missing values in each column
print("Missing values in each column:")
print(data.isnull().sum())

# fill missing values in Sleep Disorder with 'None'
data["Sleep Disorder"] = data["Sleep Disorder"].fillna("None")

# verify
print(data["Sleep Disorder"].value_counts())

# check duplicates
print("Number of duplicate rows:", data.duplicated().sum())

# check BMI category counts before cleaning
print("BMI Category before cleaning:")
print(data["BMI Category"].value_counts())

# merge 'Normal Weight' into 'Normal'
data["BMI Category"] = data["BMI Category"].replace("Normal Weight", "Normal")

# check BMI category counts after cleaning
print("\nBMI Category after cleaning:")
print(data["BMI Category"].value_counts())

print("Final dataset info after cleaning:")
data.info()

print("\nFinal missing values:")
print(data.isnull().sum())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(data["Sleep Duration"], bins=15, edgecolor="black")
plt.title("Distribution of Sleep Duration")
plt.xlabel("Sleep Duration (hours)")
plt.ylabel("Frequency")
plt.show()


df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Descriptive statistics
group_stats = df.groupby("BMI Category")["Sleep Duration"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
).round(3)

print(group_stats)

from scipy import stats

# Anova Test
normal_sleep = df[df["BMI Category"] == "Normal"]["Sleep Duration"]
overweight_sleep = df[df["BMI Category"] == "Overweight"]["Sleep Duration"]
obese_sleep = df[df["BMI Category"] == "Obese"]["Sleep Duration"]

anova_result = stats.f_oneway(normal_sleep, overweight_sleep, obese_sleep)

print(anova_result)

print("F-statistic:", anova_result.statistic)
print("p-value:", anova_result.pvalue)

if anova_result.pvalue < 0.05:
    print("Reject H0: Sleep duration is significantly different across BMI categories.")
else:
    print("Fail to reject H0: No significant difference in sleep duration across BMI categories.")

# Tukey Post-Hoc Test
tukey = pairwise_tukeyhsd(
    endog=df["Sleep Duration"],
    groups=df["BMI Category"],
    alpha=0.05
)

print(tukey)



