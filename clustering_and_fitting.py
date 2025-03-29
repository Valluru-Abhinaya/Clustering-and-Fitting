#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file, or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    """Generate and save a relational plot."""
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='math score', y='reading score', hue='gender')
    plt.title('Relational Plot of Math vs Reading Score')
    plt.savefig('relational_plot.png')
    plt.show()
    return


def plot_categorical_plot(df):
    """Generate and save a categorical plot."""
    fig, ax = plt.subplots()
    unique_genders = df['gender'].unique()
    colors = ['lightblue', 'lightcoral']
    for i, gender in enumerate(unique_genders):
        sns.boxplot(
            x='gender', y='math score', data=df[df['gender'] == gender],
            color=colors[i], width=0.5
        )
    plt.title('Categorical Plot of Math Score by Gender')
    plt.savefig('categorical_plot.png')
    plt.show()
    return


def plot_statistical_plot(df):
    """Generate and save a histogram for statistical analysis."""
    fig, ax = plt.subplots()
    sns.histplot(df['math score'], kde=True)
    plt.title('Statistical Plot of Math Score Distribution')
    plt.savefig('statistical_plot.png')
    plt.show()
    return


def statistical_analysis(df, col: str):
    """Perform statistical analysis for a given column."""
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col])
    excess_kurtosis = ss.kurtosis(df[col], fisher=True)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the data."""
    print(df.describe())
    print(df.head())
    print(df.corr(numeric_only=True))
    return df


def writing(moments, col):
    """Display the statistical analysis results."""
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    if abs(moments[2]) < 0.5:
        skewness_type = "not skewed"
    elif moments[2] > 0:
        skewness_type = "right skewed"
    else:
        skewness_type = "left skewed"

    if moments[3] < -1:
        kurtosis_type = "platykurtic"
    elif -1 <= moments[3] <= 1:
        kurtosis_type = "mesokurtic"
    else:
        kurtosis_type = "leptokurtic"

    print(f'The data was {skewness_type} and {kurtosis_type}.')
    return


def perform_clustering(df, col1, col2):
    """Perform clustering using K-Means."""
    data1 = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data1)

    def plot_elbow_method():
        """Plot and save the elbow method graph."""
        fig, ax = plt.subplots()
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0, n_init='auto')
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, 11), inertia, marker='o')
        plt.title('Elbow Method for Optimal Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.savefig('elbow_plot.png')
        plt.show()
        return

    def one_silhouette_inertia():
        """Return a silhouette score and inertia for 3 clusters."""
        kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
        kmeans.fit(scaled_data)
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        return labels, inertia, kmeans.cluster_centers_

    plot_elbow_method()
    labels, inertia, cluster_centers = one_silhouette_inertia()

    return (
        labels, 
        data1, 
        cluster_centers[:, 0], 
        cluster_centers[:, 1], 
        cluster_centers
    )


def plot_clustered_data(labels, data1, xkmeans, ykmeans, centre_labels):
    """Plot and save clustered data."""
    plt.scatter(
        data1.iloc[:, 0], data1.iloc[:, 1], 
        c=labels, cmap='viridis', alpha=0.5
    )
    plt.scatter(xkmeans, ykmeans, c='red', marker='X', label='Cluster Centers')
    plt.title('Clustered Data')
    plt.xlabel(data1.columns[0])
    plt.ylabel(data1.columns[1])
    plt.legend()
    plt.savefig('clustering.png')
    plt.show()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression."""
    data1 = df[[col1, col2]].dropna()
    X = data1[[col1]].values.reshape(-1, 1)
    y = data1[col2].values

    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    return data1, x_range, y_pred


def plot_fitted_data(data1, x, y):
    """Plot and save the fitted data."""
    fig, ax = plt.subplots()
    plt.scatter(
        data1.iloc[:, 0], data1.iloc[:, 1], 
        alpha=0.5, label='Original Data'
    )
    plt.plot(x, y, color='red', label='Fitted Line')
    plt.title('Fitting Plot')
    plt.xlabel(data1.columns[0])
    plt.ylabel(data1.columns[1])
    plt.legend()
    plt.savefig('fitting.png')
    plt.show()
    return


def main():
    """Main function to execute all tasks."""
    df = pd.read_csv(r'data.csv')
    df = preprocessing(df)
    col = 'math score'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    clustering_results = perform_clustering(df, 'math score', 'reading score')
    plot_clustered_data(*clustering_results)
    fitting_results = perform_fitting(df, 'math score', 'reading score')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
