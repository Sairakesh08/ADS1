# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 04:20:07 2024

@author: udayp
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


def load_data():
    """
    Load the Iris dataset from scikit-learn and map target numbers to species names.
    
    Returns:
        DataFrame: The Iris dataset with species names instead of target numbers.
    """
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['species'] = data.target
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species'].map(species_map)
    return df
def plot_scatter(df):
    """
    Create a scatter plot of sepal length vs sepal width categorized by species.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset with 'species' as one of the columns.
    
    Produces:
        A scatter plot with 'sepal length (cm)' on the x-axis and 'sepal width (cm)' on the y-axis,
        colored and styled by 'species'.
    """
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', style='species', s=100, palette='bright', data=df)
    plt.title('Sepal Length vs Sepal Width by Species', fontsize=16)
    plt.xlabel('Sepal Length (cm)', fontsize=14)
    plt.ylabel('Sepal Width (cm)', fontsize=14)
    scatter.set_xticklabels(scatter.get_xticks(), fontsize=12)  # Ensure x-axis labels are easily readable
    scatter.set_yticklabels(scatter.get_yticks(), fontsize=12)  # Ensure y-axis labels are easily readable
    plt.legend(title='Species', title_fontsize='13', fontsize='12', loc='upper right')
    plt.grid(True)  # Adds grid lines for better readability
    plt.show()
def plot_bar_chart(df):
    """
    Create a bar chart showing the average sepal length for each species in the Iris dataset.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset with 'species' and 'sepal length (cm)' as columns.
    
    Produces:
        A bar chart that compares the average sepal length across different species with clear visualization.
    """
    plt.figure(figsize=(10, 6))
    avg_sepal_length = df.groupby('species')['sepal length (cm)'].mean().reset_index()
    barplot = sns.barplot(x='species', y='sepal length (cm)', data=avg_sepal_length, palette='viridis')
    plt.title('Average Sepal Length by Species', fontsize=16)
    plt.xlabel('Species', fontsize=14)
    plt.ylabel('Average Sepal Length (cm)', fontsize=14)
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=0, fontsize=12)  # Ensure labels are easily readable
    barplot.set_yticklabels([f'{x:.1f} cm' for x in barplot.get_yticks()], fontsize=12)  # Format y-axis labels for better readability
    plt.show()
def plot_box_plot(df):
    """
    Create a box plot to visualize the distribution of sepal width across different species in the Iris dataset.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset with 'species' and 'sepal width (cm)' as columns.
    
    Produces:
        A box plot illustrating the spread and outliers of sepal width for each species, with clear categorization.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='species', y='sepal width (cm)', data=df, palette='Set2')
    plt.title('Distribution of Sepal Width by Species', fontsize=16)
    plt.xlabel('Species', fontsize=14)
    plt.ylabel('Sepal Width (cm)', fontsize=14)
    plt.xticks(fontsize=12)  # Ensure category labels are easily readable
    plt.yticks(fontsize=12)  # Ensure value labels are easily readable
    plt.grid(True, linestyle='--', which='major', color='gray', alpha=0.5)  # Adds grid lines for better readability
    plt.show()
def descriptive_statistics(df):
    """
    Calculate and print descriptive statistics, including the mean, median, standard deviation,
    skewness, and kurtosis for the numeric columns in the Iris dataset.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset with numeric columns.
    
    Returns:
        tuple: A tuple containing descriptive statistics, skewness, and kurtosis of the dataframe.
    """
    # Calculate description for numeric columns only
    description = df.describe()
    print("Descriptive Statistics:\n", description)

    # Selecting only numeric columns for skewness and kurtosis
    numeric_cols = df.select_dtypes(include=[np.number])  # ensures only numeric columns are considered

    skewness = numeric_cols.skew()
    print("\nSkewness:\n", skewness)

    kurtosis = numeric_cols.kurt()
    print("\nKurtosis:\n", kurtosis)

    return description, skewness, kurtosis
def test_statistical_significance(df):
    """
    Perform ANOVA (Analysis of Variance) to test if there are statistically significant differences
    in the sepal width among different species in the Iris dataset.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset with 'species' and 'sepal width (cm)' as columns.
    
    Returns:
        tuple: F-value and p-value from the ANOVA test, providing insights into statistical significance.
    """
    try:
        # Grouping data by species and extracting sepal widths
        species_groups = [df[df['species'] == species]['sepal width (cm)'] for species in df['species'].unique()]
        
        # Performing ANOVA
        f_value, p_value = stats.f_oneway(*species_groups)
        print(f"ANOVA results: F-value = {f_value:.2f}, p-value = {p_value:.3f}")
        
    except Exception as e:
        print(f"Error in performing ANOVA: {e}")
        f_value, p_value = None, None  # Properly handle the case where ANOVA fails

    return f_value, p_value
def perform_clustering(df, n_clusters=3):
    """
    Perform K-means clustering on the dataset to identify distinct groups based on the numeric features.
    This function also visualizes the resulting clusters and evaluates them using the silhouette score.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset.
        n_clusters (int): The number of clusters to use.

    Returns:
        DataFrame: Original dataframe with an additional column for cluster labels.
        float: Silhouette score of the clustering to evaluate its effectiveness.
    """
    # Selecting numeric features for clustering
    features = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    df['cluster'] = cluster_labels

    # Calculate silhouette score
    silhouette = silhouette_score(features_scaled, cluster_labels)
    print(f"Silhouette Score: {silhouette:.2f}")

    # Visualizing the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=cluster_labels, cmap='viridis', marker='o', s=50, alpha=0.8)
    plt.title('Visualization of Clusters', fontsize=16)
    plt.xlabel('Sepal Length (cm)', fontsize=14)
    plt.ylabel('Sepal Width (cm)', fontsize=14)
    plt.colorbar(ticks=range(n_clusters), label='Cluster')
    plt.grid(True)
    plt.show()

    return df, silhouette
def perform_fitting(df):
    """
    Fit a linear regression model to predict sepal width (cm) using other features in the Iris dataset.
    This function also visualizes the fitted model against actual data points to assess fitting quality,
    and provides mean squared error as a measure of model performance.

    Parameters:
        df (DataFrame): The dataframe containing the Iris dataset.

    Returns:
        LinearRegression: The fitted model.
        float: Mean squared error of the model on test data.
    """
    # Preparing data for fitting
    X = df[['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = df['sepal width (cm)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Visualization of the fitting results
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['sepal length (cm)'], y_test, color='blue', label='Actual Data')
    plt.scatter(X_test['sepal length (cm)'], y_pred, color='red', marker='x', label='Predicted Data')
    plt.title('Fit Visualization: Actual vs Predicted Sepal Width', fontsize=16)
    plt.xlabel('Sepal Length (cm)', fontsize=14)
    plt.ylabel('Sepal Width (cm)', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, mse

def plot_correlation_matrix(df):
    """
    Plot a correlation matrix for the numeric features of the Iris dataset.
    
    Parameters:
        df (DataFrame): The dataframe containing the Iris data.
    """
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    return correlation_matrix
def main():
    """
    Main function to load data, perform analyses, and visualize results.

    Executes the following steps:
    1. Load and preprocess the data.
    2. Perform descriptive and inferential statistics.
    3. Conduct clustering and evaluate its quality.
    4. Perform regression fitting and visualize fitting quality.
    5. Visualize various aspects of the data using different types of plots.
    """
    # Load the data
    df = load_data()
    print("Data loaded successfully.")

    # Perform and display descriptive statistics
    descriptive_stats = descriptive_statistics(df)
    print("Descriptive statistics completed.")

    # Perform and display results of inferential statistics
    f_value, p_value = test_statistical_significance(df)
    print("Inferential statistics analysis completed with ANOVA results.")

    # Clustering
    df, silhouette = perform_clustering(df)
    print("Cluster analysis completed with Silhouette Score: {:.2f}".format(silhouette))

    # Fitting
    model, mse = perform_fitting(df)
    print("Fitting completed with Mean Squared Error: {:.2f}".format(mse))

    # Plotting
    print("Displaying scatter plot...")
    plot_scatter(df)

    print("Displaying bar chart...")
    plot_bar_chart(df)

    print("Displaying box plot...")
    plot_box_plot(df)

    print("Displaying correlation matrix...")
    plot_correlation_matrix(df)

    print("Analysis completed successfully. Review the statistics and plots above.")

if __name__ == "__main__":
    main()
