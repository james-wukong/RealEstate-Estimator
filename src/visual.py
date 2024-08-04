from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.constants import GrpColumns


class DataVisual:
    def __init__(self) -> None:
        pass

    def value_boxplot(
        self,
        dataframe: pd.DataFrame,
        cols: list[str] | str | bool,
    ) -> None:
        """pass in numerical features and create box plot for each feature

        Args:
            cols (list[str]): _description_
        """
        if isinstance(cols, str):
            cols = [cols]
        elif isinstance(cols, bool):
            cols = dataframe.select_dtypes(include=[np.number])

        if cols:
            # Loop through features and create boxplots
            for col in cols:
                plt.figure()  # Create a new figure for each boxplot
                dataframe[col].plot.box()
                plt.title(col)
                plt.show()

    def value_dist(
        self,
        dataframe: pd.DataFrame,
        cols: list[str] | str | bool,
    ) -> None:
        """pass in numerical features and create the distribution for each

        Args:
            cols (str | list[str]): _description_
        """
        if isinstance(cols, str):
            cols = [cols]
        elif isinstance(cols, bool):
            cols = dataframe.select_dtypes(include=[np.number])

        if cols:
            # Loop through numerical features and create KDE plots
            # for col in df.select_dtypes(include=[np.number]):
            for col in cols:
                plt.figure()  # Create a new figure for each KDE plot
                sns.kdeplot(data=dataframe[col], fill=True)
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.title(f"Distribution of {col} (KDE)")
                plt.show()

    def bar_chart(
        self,
        dataframe: pd.DataFrame,
        cols: list[str] | str | bool,
    ) -> None:
        """plot boolean type of values in bar chart

        Args:
            dataframe (pd.DataFrame): _description_
            cols (list[str] | str): _description_
        """
        if isinstance(cols, str):
            cols = [cols]
        elif isinstance(cols, bool):
            cols = dataframe.select_dtypes(include=[bool])

        if cols:
            # Get value counts for boolean features
            # for col in dataframe.select_dtypes(include=[bool]):
            for col in cols:
                value_counts = dataframe[
                    col
                ].value_counts()  # Count True and False occurrences

                # Create a bar chart
                plt.figure()
                value_counts.plot(
                    kind="bar"  # , color=["blue", "orange"]
                )  # Colors for True/False
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.title(f"Distribution of {col} (Value Counts)")
                # plt.xticks(
                #     [0, 1], ["False", "True"]
                # )  # Set custom x-axis labels for clarity
                plt.show()

    def three_chart_plot(self, df: pd.DataFrame, feature: str) -> None:
        fig = plt.figure(constrained_layout=True, figsize=(12, 12))
        grid = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

        ax1 = fig.add_subplot(grid[0, :2])
        ax1.set_title("Histogram")

        # sns.distplot(df.loc[:, feature], norm_hist=True, ax=ax1)
        sns.histplot(data=df, x=feature, kde=True, ax=ax1)
        # sns.displot(data=df, y=feature, kind="hist", ax=ax1)
        plt.axvline(x=df[feature].mean(), c="red")
        plt.axvline(x=df[feature].median(), c="green")

        ax2 = fig.add_subplot(grid[1, :2])
        ax2.set_title("QQ_plot")
        stats.probplot(df.loc[:, feature], plot=ax2)

        # Customizing the Box Plot.
        ax3 = fig.add_subplot(grid[:, 2])
        # Set title.
        ax3.set_title("Box Plot")
        sns.boxplot(df.loc[:, feature], orient="v", ax=ax3)

    def show_miss_percent(
        self,
        data: pd.DataFrame,
        thresh: int = 20,
        color: str = sns.color_palette("Reds", 15),
        edgecolor: str = "black",
        height: int = 6,
        width: int = 25,
    ) -> None:
        if GrpColumns.Y_COL in data.columns:
            df = data.drop(columns=GrpColumns.Y_COL)
        plt.figure(figsize=(width, height))
        percentage = (df.isnull().mean()) * 100
        percentage.sort_values(ascending=False).plot.bar(
            color=color, edgecolor=edgecolor
        )
        plt.axhline(y=thresh, color="r", linestyle="-")

        plt.title(
            "Missing values percentage per column",
            fontsize=20,
            weight="bold",
        )

        plt.text(
            len(df.isnull().sum() / len(df)) / 1.7,
            thresh + 12.5,
            f"Columns with more than {thresh}% missing values",
            fontsize=12,
            color="crimson",
            ha="left",
            va="top",
        )
        plt.text(
            len(data.isnull().sum() / len(df)) / 1.7,
            thresh - 5,
            f"Columns with less than {thresh}% missing values",
            fontsize=12,
            color="green",
            ha="left",
            va="top",
        )
        plt.xlabel("Columns", size=15, weight="bold")
        plt.ylabel("Missing values percentage")
        plt.yticks(weight="bold")

        return plt.show()

    def plot_pca(
        self,
        df: pd.DataFrame,
        n_comp: int = 10,
        k_cluster: int = 10,
        plot: bool = False,
    ) -> None:
        if GrpColumns.Y_COL in df.columns:
            # df = data.drop(columns=GrpColumns.Y_COL)
            df = df.drop(columns=GrpColumns.Y_COL, axis=0)
        # Generate some sample data
        model = TSNE(n_components=2, random_state=0, perplexity=50)
        tsne = model.fit_transform(df)

        # Standardize the data
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # Apply PCA
        pca = PCA(n_components=n_comp)  # Choose the number of components
        df_pca = pca.fit_transform(df_scaled)

        kmeans = KMeans(n_clusters=k_cluster)
        kmeans.fit(df_pca)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        # print("Explained variance ratio:", explained_variance)

        # Create a DataFrame with the principal components
        df_pca = pd.DataFrame(
            df_pca,
            columns=[f"PCA_{i+1}" for i in range(n_comp)],
        )

        # df_pca now contains the reduced set of features
        # print(df_pca.head())

        if plot:
            plt.plot(np.cumsum(explained_variance))
            plt.xlabel("number of components")
            plt.ylabel("cumulative explained variance")

            fr = pd.DataFrame(
                {
                    "tsne1": tsne[:, 0],
                    "tsne2": tsne[:, 1],
                    "cluster": kmeans.labels_,
                }
            )
            sns.lmplot(
                data=fr,
                x="tsne1",
                y="tsne2",
                hue="cluster",
                fit_reg=False,
            )

    def scatter_y_and_yhat(self, y: Iterable, yhat: np.ndarray) -> None:
        y = np.expm1(y)
        yhat = np.expm1(yhat)
        plt.figure(figsize=(10, 6))
        plt.scatter(y, yhat, alpha=0.5)
        plt.plot(
            [min(y), max(y)],
            [min(y), max(y)],
            color="red",
            linestyle="--",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.show()
