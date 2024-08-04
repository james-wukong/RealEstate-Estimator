import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from src.constants import GrpColumns


def corr_selector(df: pd.DataFrame, corr_th: float = 0.75) -> list[str]:
    # Get the column names of the DataFrame
    matrix = df.corr(method="pearson").abs()
    columns = matrix.columns

    # Create an empty list to keep track of columns to drop
    columns_to_drop = []

    # Loop over the columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            # Access the cell of the DataFrame
            if matrix.loc[columns[i], columns[j]] > corr_th:
                columns_to_drop.append(columns[j])

    return columns_to_drop


def high_correlated_cols(
    dataframe: pd.DataFrame, plot: bool = False, corr_th: float = 0.75
):
    if GrpColumns.Y_COL in dataframe.columns:
        # df = data.drop(columns=GrpColumns.Y_COL)
        dataframe = dataframe.drop(columns=GrpColumns.Y_COL, axis=0)
    # numeric_df = dataframe.select_dtypes(include=[np.number])
    corr = dataframe.corr(method="pearson")
    print("len of columns", len(corr.columns))
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool)
    )
    drop_list = [
        col
        for col in upper_triangle_matrix.columns
        if any(upper_triangle_matrix[col] > corr_th)
    ]
    if plot:
        sns.set_theme(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list
