# Data Analysis and Visualization with Pandas and Matplotlib
# Assignment: Analyzing the Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better-looking plots
sns.set(style="whitegrid")

def main():
    """
    Main function to load, explore, analyze, and visualize the Iris dataset.
    """
    print("=== Loading and Exploring the Iris Dataset ===\n")

    try:
        # Load the Iris dataset using sklearn
        iris = load_iris()
        
        # Create DataFrame with feature names and target species
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}\n")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Task 1: Explore the Dataset
    print("=== Task 1: Data Exploration ===")
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")

    print("Dataset info (data types and non-null counts):")
    print(df.info(), "\n")

    print("Missing values in each column:")
    print(df.isnull().sum(), "\n")

    # No missing values in Iris dataset, but let's simulate cleaning just in case
    # (for demonstration purposes)
    df_cleaned = df.dropna()  # No actual drops needed here
    print("✅ No missing values found. Dataset is clean.\n")

    # Task 2: Basic Data Analysis
    print("=== Task 2: Basic Data Analysis ===")
    print("Basic statistics of numerical columns:")
    print(df.describe(), "\n")

    # Group by species and compute mean of numerical features
    grouped_means = df.groupby('species').mean()
    print("Mean values per species:")
    print(grouped_means, "\n")

    # Interesting finding: Species have very distinct average petal lengths
    max_petal_species = grouped_means['petal length (cm)'].idxmax()
    min_petal_species = grouped_means['petal length (cm)'].idxmin()
    print(f"💡 Insight: '{max_petal_species}' has the highest average petal length ({grouped_means['petal length (cm)'].max():.2f} cm)")
    print(f"💡 Insight: '{min_petal_species}' has the lowest average petal length ({grouped_means['petal length (cm)'].min():.2f} cm)\n")

    # Task 3: Data Visualization
    print("=== Task 3: Data Visualization ===")
    
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Iris Dataset Analysis: Visualizations", fontsize=16, fontweight='bold')

    # 1. Line Chart: Trends over time (Simulated - since Iris has no time dimension)
    # We'll create a synthetic time series by ordering samples by petal length
    df_sorted = df.sort_values('petal length (cm)').reset_index(drop=True)
    df_sorted['sample_index'] = df_sorted.index + 1
    axes[0, 0].plot(df_sorted['sample_index'], df_sorted['petal length (cm)'], color='blue', linewidth=1.5, label='Petal Length')
    axes[0, 0].set_title("Line Chart: Petal Length Trend (Ordered by Size)")
    axes[0, 0].set_xlabel("Sample Index (Ordered by Petal Length)")
    axes[0, 0].set_ylabel("Petal Length (cm)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Bar Chart: Average petal length per species
    species_means = df.groupby('species')['petal length (cm)'].mean()
    bars = axes[0, 1].bar(species_means.index, species_means.values, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_title("Bar Chart: Average Petal Length by Species")
    axes[0, 1].set_xlabel("Species")
    axes[0, 1].set_ylabel("Average Petal Length (cm)")
    axes[0, 1].tick_params(axis='x', rotation=15)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    # 3. Histogram: Distribution of sepal width
    axes[1, 0].hist(df['sepal width (cm)'], bins=15, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("Histogram: Distribution of Sepal Width")
    axes[1, 0].set_xlabel("Sepal Width (cm)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].axvline(df['sepal width (cm)'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["sepal width (cm)"].mean():.2f}')
    axes[1, 0].legend()

    # 4. Scatter Plot: Sepal Length vs Petal Length
    scatter = axes[1, 1].scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                                 c=df['species'].cat.codes, cmap='viridis', alpha=0.7, s=60)
    axes[1, 1].set_title("Scatter Plot: Sepal Length vs Petal Length")
    axes[1, 1].set_xlabel("Sepal Length (cm)")
    axes[1, 1].set_ylabel("Petal Length (cm)")
    # Add legend for species
    legend_labels = iris.target_names
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    axes[1, 1].legend(handles, legend_labels, title="Species", loc="upper left")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.show()

    # Optional: Print final observations
    print("\n=== Final Observations ===")
    print("1. The Iris dataset has no missing values — ready for analysis.")
    print("2. Petal length is the most discriminative feature across species.")
    print("3. Setosa has significantly smaller petals than Versicolor and Virginica.")
    print("4. Sepal width shows a roughly normal distribution.")
    print("5. There is a strong positive correlation between sepal length and petal length (visible in scatter plot).")
    print("6. All visualizations clearly separate the three species, making this dataset ideal for classification tasks.")

if __name__ == "__main__":
    main()