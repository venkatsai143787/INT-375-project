# 1. Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# 2. Load & Preprocess Data

df = pd.read_csv("Public_Libraries.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert to numeric (invalid → NaN)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Keep only numeric columns
numeric_df = df.select_dtypes(include=np.number)

# Remove empty columns
numeric_df = numeric_df.dropna(axis=1, how='all')

# Fill missing values
for col in numeric_df.columns:
    numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())

cols = numeric_df.columns

print("Columns used:", cols)
print(numeric_df.head())


# 3. Visualizations

#  Histogram
if len(cols) > 0:
    plt.figure()
    plt.hist(numeric_df[cols[0]], bins=15)
    plt.title("Histogram")
    plt.xlabel(cols[0])
    plt.ylabel("Frequency")
    plt.show()


# Bar Chart
if len(cols) > 0:
    plt.figure()
    numeric_df[cols[0]].head(10).plot(kind='bar')
    plt.title("Bar Chart")
    plt.show()


# Line Chart
if len(cols) >= 2:
    plt.figure()
    sorted_df = numeric_df.sort_values(by=cols[0])
    plt.plot(sorted_df[cols[0]], sorted_df[cols[1]])
    plt.title("Line Chart")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()

# Pie Chart (using bins)
if len(cols) > 0:
    plt.figure()
    
    # Convert numeric data into categories
    binned = pd.cut(numeric_df[cols[0]], bins=5)
    
    binned.value_counts().plot(kind='pie', autopct='%1.1f%%')
    
    plt.title("Pie Chart (Binned Data)")
    plt.ylabel("")
    plt.show()


#  Box Plot
if len(cols) > 0:
    plt.figure()
    sns.boxplot(x=numeric_df[cols[0]])
    plt.title("Box Plot")
    plt.show()


#  Scatter Plot
if len(cols) >= 2:
    plt.figure()
    plt.scatter(numeric_df[cols[0]], numeric_df[cols[1]])
    plt.title("Scatter Plot")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()


#  Bubble Plot
if len(cols) >= 3:
    plt.figure()
    plt.scatter(
        numeric_df[cols[0]],
        numeric_df[cols[1]],
        s=numeric_df[cols[2]] * 0.1,
        alpha=0.5
    )
    plt.title("Bubble Plot")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()


#  Heatmap
plt.figure(figsize=(6,4))
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.show()


#  Pairplot (limited columns to avoid congestion)
selected_cols = cols[:4] if len(cols) >= 4 else cols
sns.pairplot(numeric_df[selected_cols])
plt.show()


# 4. Statistical Modeling

if len(cols) >= 2:
    corr_value = corr_matrix.iloc[0, 1]
    print("\nCorrelation between first two variables:", corr_value)

    if abs(corr_value) > 0.5:
        print("Strong relationship")
    elif abs(corr_value) > 0.3:
        print("Moderate relationship")
    else:
        print("Weak relationship")


# 5. Machine Learning Model

if len(cols) >= 2:
    X = numeric_df[[cols[0]]]
    y = numeric_df[cols[1]]

    # Remove any remaining NaN
    data = pd.concat([X, y], axis=1).dropna()
    X = data[[cols[0]]]
    y = data[cols[1]]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Results:")
    print("R2 Score:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))


# 6. Final Regression Graph

    plt.figure()
    plt.scatter(X, y)

    sorted_df = data.sort_values(by=cols[0])

    plt.plot(sorted_df[cols[0]],
             model.predict(sorted_df[[cols[0]]]),
             color='red')

    plt.title("Linear Regression Model")
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.show()
