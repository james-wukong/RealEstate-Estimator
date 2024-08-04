import os
import pathlib
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from dateutil.relativedelta import relativedelta
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import PowerTransformer

# from scipy.stats import skew

from src.constants import FreqConst, GrpColumns


class DataClean:
    __dataframe = pd.DataFrame()

    def __init__(
        self,
        file: str | None,
        low_mem: bool = True,
        usecolds: list | None = None,
        out_file: str = "result.csv",
    ) -> None:
        """initalize fields

        Args:
            file (str | None): filename that contains data to be processed
            low_mem (bool): low memory settings for pandas dataframe
            usecolds (list | None): use specified cols only
        """
        self.__file = file
        self.__out_file = out_file
        self.__removed_cols = []
        self.__skew_params = {}
        self.__label_encoders = {}
        self.y = GrpColumns.Y_COL
        if file:
            if pathlib.Path(file).suffix.lower() == ".csv":
                self.__dataframe = pd.read_csv(
                    file,
                    low_memory=low_mem,
                    skipinitialspace=True,
                    usecols=usecolds,
                )
            elif pathlib.Path(file).suffix.lower() in [".xls", ".xlsx"]:
                self.__dataframe = pd.read_excel(
                    file,
                    low_memory=low_mem,
                    skipinitialspace=True,
                    usecols=usecolds,
                )
            else:
                self.__dataframe = pd.DataFrame()

    @property
    def file(self) -> str:
        return self.__file

    @file.setter
    def file(self, filename: str) -> None:
        self.__file = filename

    @property
    def out_file(self) -> str:
        return self.__out_file

    @out_file.setter
    def out_file(self, value) -> None:
        self.__out_file = value

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.__dataframe

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame) -> None:
        self.__dataframe = df

    @property
    def removed_cols(self) -> list[str]:
        return self.__removed_cols

    @removed_cols.setter
    def removed_cols(self, value: str | list[str]) -> None:
        self.__removed_cols.extend(value)

    def export_csv(self) -> None:
        """save the result csv file for data"""
        outfile = os.path.join("data", "csv", self.out_file)
        self.dataframe.to_csv(outfile)

    def remove_missing(
        self,
        miss_perc: float = 100,
        unique_th: int = 15,
    ) -> None:
        """
        1. check missing values in dataset
        2. remove features that are miss_perc missing
        """
        if not self.dataframe.empty:
            rm_cols = []
            for i, col in enumerate(self.dataframe.columns):
                missing_data = self.dataframe.iloc[:, i].isna().sum()
                perc = missing_data / len(self.dataframe) * 100
                if perc - miss_perc >= 0:
                    rm_cols.append(col)
                    print(f"Removing feature: {col} with {perc}% missing")
                else:
                    print(
                        f"Feature {col} >> Missing entries: {missing_data} "
                        f"|  Percentage: {round(perc, 2)}"
                        f" | Unique values: "
                        f"{len(self.dataframe[col].unique())}"
                    )
                    if len(self.dataframe[col].unique()) <= unique_th:
                        print(self.dataframe[col].unique())
            # if cols not empty, remove all features in cols
            self.drop_cols(cols=rm_cols)
            print(f"Total features removed: {len(rm_cols)}")
            print(f"Removed cols: {rm_cols}")

    def check_duplicated(self) -> None:
        """
        1. check duplicated rows
        2. remove duplicated rows if exist
        """
        if not self.dataframe.empty and self.dataframe.duplicated().sum() > 0:
            self.dataframe.drop_duplicates(inplace=True)

    def check_unique_data(self, cols: list[str]) -> None:
        """check unique values in cols

        Args:
            cols (list[str]): _description_
        """
        if cols:
            for col in cols:
                print(
                    f"dataset {col} uniques: "
                    f"{self.dataframe[col].astype(str).unique()}"
                )

    def check_numeric_data(self, cols: dict[str, list[str]]) -> None:
        """describe numeric columns

        Args:
            cols (dict[str, list[str]]): _description_
        """
        if cols:
            for _, num_cols in cols:
                for col in num_cols:
                    print(f"{col}: {self.dataframe[col].describe()}")

    def drop_cols(self, cols: list[str]) -> None:
        """drop irrelevant columns and update dataframe

        Args:
            drop_cols (list): list of columns that are going to be dropped
        """
        if set(cols).issubset(set(self.dataframe.columns)):
            self.dataframe.drop(columns=list(set(cols)), axis=1, inplace=True)
            self.removed_cols = cols
        else:
            rm_cols = set(cols) & set(self.dataframe.columns)
            self.dataframe.drop(columns=list(rm_cols), axis=1, inplace=True)
            self.removed_cols = rm_cols

    def drop_dirty_rows(self) -> None:
        uuid_pattern = re.compile(
            r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
            re.IGNORECASE,
        )
        self.dataframe = self.dataframe[
            self.dataframe["id"].str.match(uuid_pattern, na=False)
        ].reset_index(drop=True)

    def replace_empty_brackets(self) -> None:
        """replace empty brackets, such as "{}" or "{None}"
        into np.nan, so that it's easier to know the missing percentage
        """
        self.dataframe.replace(
            to_replace=["{}", "{None}", "", "N/A"],
            value=np.nan,
            inplace=True,
        )

    def fill_median(self, cols: list[str]) -> None:
        if cols:
            self.__dataframe[cols].fillna(
                self.__dataframe[cols].median(),
                inplace=True,
            )

    def fill_zero(self, cols: list[str]) -> None:
        if cols:
            self.__dataframe[cols].fillna(0, inplace=True)

    def parse_categorical_cols(self, value: str) -> list[str]:
        """clean categorical cols data

        Args:
            value (str): _description_

        Returns:
            list[str]: _description_
        """
        if isinstance(value, float):
            value = str(value)

        value = re.sub(r"[^a-zA-Z0-9,-\/]", "", value)
        return value.split(",")
        # return value

    def conv_category_cols(self, cols: list[str]) -> None:
        # Parse categorical cols
        self.__dataframe[cols] = self.__dataframe[cols].map(
            self.parse_categorical_cols,
        )

    def mlb_transform_cols(self, cols: list[str]) -> None:
        """transform cols into multilabel encoded columns
        update the dataframe property
        Args:
            cols (list[str]): cols to be tranformed

        Returns:
            None:
        """
        mlb = MultiLabelBinarizer()
        self.conv_category_cols(cols)

        # Function to apply MultiLabelBinarizer to a column
        def mlb_transform_column(col: str) -> pd.DataFrame:
            df = pd.DataFrame(
                mlb.fit_transform(self.__dataframe[col]),
                columns=mlb.classes_,
                index=self.__dataframe[col].index,
            )
            # Save the fitted mlb to a file
            with open(
                os.path.join("app/saved-models/mlb", col + "_mlb.sav"),
                "wb",
            ) as file:
                joblib.dump(mlb, file)
            return df

        for col in cols:
            # Apply MultiLabelBinarizer to each column
            enc_col = mlb_transform_column(
                col,
            ).astype(np.bool_)
            # Rename columns to avoid collision
            enc_col.columns = [f"{col.lower()}_{c.lower()}" for c in enc_col.columns]
            # print(enc_col.columns)
            self.dataframe = pd.concat([self.dataframe, enc_col], axis=1)
            # drop the empty column
            if self.__dataframe.get(f"{col}_") is not None:
                self.drop_cols([f"{col}_"])
        # drop original cols
        self.drop_cols(cols)

    def convert_dtypes(self, cols: list[str], to: np.dtype) -> None:
        self.__dataframe[cols] = self.__dataframe[cols].astype(to)

    def update_dtypes(self) -> None:
        # convert date types
        for d in GrpColumns.DATE_COLS:
            self.dataframe[d] = pd.to_datetime(self.dataframe[d])
        # convert bool types
        self.convert_dtypes(
            cols=(
                GrpColumns.BOOL_COLS["false"]
                + GrpColumns.BOOL_COLS["freq"]
                + GrpColumns.BOOL_COLS["other"]
            ),
            to=np.bool_,
        )
        # convert integer type
        self.convert_dtypes(
            cols=(
                GrpColumns.INT_COLS["median"]
                + GrpColumns.INT_COLS["zero"]
                + GrpColumns.INT_COLS["freq"]
                + GrpColumns.INT_COLS["other"]
            ),
            to=np.int32,
        )
        # convert float type
        self.convert_dtypes(
            cols=(
                GrpColumns.FLOAT_COLS["median"]
                + GrpColumns.FLOAT_COLS["zero"]
                + GrpColumns.FLOAT_COLS["other"]
            ),
            to=np.float32,
        )

    def merg_association_cols(
        self,
        freq_cols: list[str],
        fee_cols: list[str],
        new_col: str,
    ) -> None:
        """merge association_fee and association_fee_frequency cols
        by creating new cols: association_fee_annual
        and dropping association_fee and association_fee_frequency cols
        """
        for freq_col in freq_cols:
            self.__dataframe[freq_col] = self.__dataframe[freq_col].map(
                {
                    "Monthly": FreqConst.MONTHLY,
                    "Quarterly": FreqConst.QUARTERLY,
                    "Semi-Annually": FreqConst.SEMI_ANNUALLY,
                    "Annually": FreqConst.ANNUALLY,
                    np.nan: FreqConst.NAN,
                }
            )
        for fee_col in fee_cols:
            self.__dataframe[fee_col].fillna(0, inplace=True)
            self.__dataframe[fee_col] = pd.to_numeric(
                self.__dataframe[fee_col], errors="coerce"
            )
        self.__dataframe[new_col] = sum(
            self.__dataframe[freq_cols[i]] * self.__dataframe[fee_cols[i]]
            for i in range(len(freq_cols))
        )
        self.__dataframe[new_col] = self.__dataframe[new_col].astype(
            np.float32,
        )
        self.drop_cols(cols=freq_cols + fee_cols)

    def update_binary_cols(self, cols: dict[str, list[str]]) -> None:
        """convert binary cols into bool data type
        fill in empty values with False

        Args:
            cols (list[str]): _description_
        """
        mapping = {
            "yes": True,
            "ok": True,
            "1": True,
            "true": True,
            "1.0": True,
            "participant options": True,
            "furnished": True,
        }
        for key, col in cols.items():
            if key == "false":
                # Apply the mapping and fill NaN values with False
                for c in col:
                    self.__dataframe[c] = (
                        self.__dataframe[c]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(mapping)
                        .fillna(False)
                        .astype(bool)
                    )
            elif key == "freq":
                # Apply the mapping and fill NaN values
                # with the most frequently
                for c in col:
                    self.__dataframe[c] = (
                        self.__dataframe[c]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(mapping)
                        .fillna(self.__dataframe[col].mode().iloc[0])
                        .astype(bool)
                    )
                    print(
                        "for col",
                        c,
                        "fill na median",
                        self.__dataframe[c].median(),
                    )

    def update_num_cols(self, cols: dict[str, list[str]]) -> None:
        """convert numerical cols into float32
        fillin missing values accordingly

        Args:
            cols (dict[str, list[str]]): _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "median":
                    for col in values:
                        # Convert to numeric and coerce errors to NaN
                        self.__dataframe[col] = pd.to_numeric(
                            self.__dataframe[col], errors="coerce"
                        )
                        # Fill NaN values with median
                        self.__dataframe[col] = (
                            self.__dataframe[col]
                            .fillna(self.__dataframe[col].median())
                            .astype(
                                np.float32,
                            )
                        )
                        print(
                            "for col",
                            col,
                            "fill na median",
                            self.__dataframe[col].median(),
                        )
                elif key == "zero":
                    for col in values:
                        if col in self.__dataframe.columns:
                            self.__dataframe[col] = (
                                self.__dataframe[col]
                                .fillna(0)
                                .astype(
                                    np.float32,
                                )
                            )
                elif key == "other":
                    self.__dataframe[values] = self.__dataframe[values].astype(
                        np.float32,
                    )

    def update_int_cols(self, cols: dict[str, list[str]]) -> None:
        """convert int cols into int32
        fillin missing values accordingly

        Args:
            cols (dict[str, list[str]]): _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "median":
                    for col in values:
                        self.__dataframe[col] = (
                            self.__dataframe[col]
                            .fillna(self.__dataframe[col].median())
                            .astype(np.int32)
                        )
                        print(
                            "for col",
                            col,
                            "fill na median",
                            self.__dataframe[col].mode()[0],
                        )
                elif key == "zero":
                    for col in values:
                        print("processing", col)
                        self.__dataframe[col] = (
                            self.__dataframe[col]
                            .fillna(0)
                            .astype(
                                np.int32,
                            )
                        )
                        print("processing done", col)
                elif key == "freq":
                    for col in values:
                        # Calculate the mode of the feature and fill NaN values
                        if col == "year_built":
                            self.__dataframe.replace(
                                {col: 0},
                                np.nan,
                                inplace=True,
                            )
                        self.__dataframe[col] = (
                            self.__dataframe[col]
                            .fillna(self.__dataframe[col].mode().iloc[0])
                            .astype(np.int32)
                        )
                        print(
                            "for col",
                            col,
                            "fill na freq",
                            self.__dataframe[col].mode()[0],
                        )
                elif key == "other":
                    self.__dataframe[values] = self.__dataframe[values].astype(
                        np.int32,
                    )

    def update_str_cols(self, cols: dict[str, list[str]]) -> None:
        """convert int cols into string
        fillin missing values accordingly

        Args:
            cols (dict[str, list[str]]): _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "freq":
                    for col in values:
                        # Calculate the mode of the feature and fill NaN values
                        self.__dataframe[col] = (
                            self.__dataframe[col]
                            .fillna(self.__dataframe[col].mode().iloc[0])
                            .astype(str)
                        )
                        print(
                            "for col",
                            col,
                            "fill na freq",
                            self.__dataframe[col].mode()[0],
                        )
                elif key in ["Unspecified", "None"]:
                    self.__dataframe[values] = (
                        self.__dataframe[values].fillna(key).astype(str)
                    )
                elif key == "other":
                    self.__dataframe[values] = self.__dataframe[values].astype(
                        str,
                    )

    def update_association_yn(
        self,
        fields: list[str],
        target_col: str,
    ) -> None:
        # Modify target_col based on condition (any field in fields not empty)
        def update_target_col(row):
            return any(pd.notna(row[field]) for field in fields)

        self.dataframe[target_col] = self.dataframe.apply(
            update_target_col,
            axis=1,
        ).astype(np.bool_)

    def update_dependant_yn(self, depends: str, target: str) -> None:
        self.dataframe[target] = self.dataframe[target].map(
            {"TRUE": True, "FALSE": False}
        )
        self.dataframe[target] = np.where(
            self.dataframe[target].isna(),
            self.dataframe[depends].notna(),
            self.dataframe[target],
        ).astype(np.bool_)

    def extract_date(self, cols: list[str]) -> None:
        def months_between(start_date, end_date):
            delta = relativedelta(end_date, start_date)
            return delta.years * 12 + delta.months

        for col in cols:
            self.dataframe[col] = pd.to_datetime(self.dataframe[col])
            self.__dataframe[col].fillna(
                self.__dataframe[col].mode()[0],
                inplace=True,
            )
            print(
                "for col",
                col,
                "fill na freq",
                self.__dataframe[col].mode()[0],
            )
            self.dataframe[col + "_year"] = self.dataframe[col].dt.year
            self.dataframe[col + "_month"] = self.dataframe[col].dt.month
            self.dataframe[col + "_day"] = self.dataframe[col].dt.day
        self.dataframe["months_between_list"] = self.dataframe.apply(
            lambda row: months_between(
                row["original_entry_timestamp"], row["listing_contract_date"]
            ),
            axis=1,
        )
        today = pd.Timestamp("today")
        self.dataframe["months_till_today"] = self.dataframe[
            "original_entry_timestamp"
        ].apply(lambda start_date: months_between(start_date, today))

        self.drop_cols(cols=cols)

    # def target_skew(self) -> str:
    #     skew = self.dataframe[[self.y]].skew()
    #     return f"Skewness of the SalesPrice is {skew}"

    def target_enc(
        self,
        cols: list[str],
        min_samples_leaf=20,
        smoothing=10,
    ) -> None:
        kf = KFold(n_splits=10, shuffle=True, random_state=100)

        for col in cols:
            self.dataframe[f"{col}_enc"] = np.nan
            for train_idx, val_idx in kf.split(self.dataframe):
                train_idx = self.dataframe.index[train_idx]
                val_idx = self.dataframe.index[val_idx]
                encoder = TargetEncoder(
                    cols=[col],
                    smoothing=smoothing,
                    min_samples_leaf=min_samples_leaf,
                )
                encoder.fit_transform(
                    self.dataframe.loc[train_idx, col],
                    self.dataframe.loc[train_idx, self.y],
                )

                self.dataframe.loc[
                    val_idx,
                    f"{col}_enc",
                ] = encoder.transform(
                    self.dataframe.loc[val_idx, [col]]
                )[col]
                # Save the fitted encoder to a file
                with open(
                    os.path.join("app/saved-models/enc", col + "_enc.sav"),
                    "wb",
                ) as file:
                    joblib.dump(encoder, file)

        self.drop_cols(cols=cols)

    def one_hot_enc(self, cols: list[str]) -> None:
        all_dummies = []
        for col in cols:
            dummies = pd.get_dummies(
                self.dataframe[col],
                drop_first=True,
                prefix=col,
            ).reset_index(drop=True)
            all_dummies.extend(dummies.columns)
            self.dataframe = pd.concat(
                [self.dataframe, dummies],
                axis=1,
            ).reset_index(drop=True)
        with open(
            os.path.join("app/saved-models/enc/one_hot_enc.sav"),
            "wb",
        ) as file:
            joblib.dump(all_dummies, file)
        self.drop_cols(cols=cols)

    def label_enc(self, cols: list[str]) -> None:
        for col in cols:
            label_encoder = LabelEncoder()
            self.dataframe[col + "_enc"] = label_encoder.fit_transform(
                self.dataframe[col]
            )
            self.__label_encoders[col] = label_encoder
        # Save the encoders
        joblib.dump(
            self.__label_encoders,
            "app/saved-models/enc/label_enc.sav",
        )
        self.drop_cols(cols=cols)

    def nplog_y(self) -> None:
        self.dataframe[self.y] = np.log1p(self.dataframe[self.y])

    def update_skewed(self) -> None:
        features = list(
            (
                set(GrpColumns.INT_COLS["median"])
                | set(GrpColumns.INT_COLS["zero"])
                | set(GrpColumns.INT_COLS["freq"])
                | set(GrpColumns.INT_COLS["other"])
                | set(GrpColumns.FLOAT_COLS["median"])
                | set(GrpColumns.FLOAT_COLS["zero"])
            )
            - set(["year_built", "original_list_price"])
        )
        skew_features = (
            self.dataframe[features]
            .apply(lambda x: skew(x))
            .sort_values(ascending=False)
        )

        high_skew = skew_features[skew_features > 1.5]
        skew_index = high_skew.index
        # print(skew_index)

        for i in skew_index:
            try:
                # Check for negative or zero values
                if (self.dataframe[i] <= 0).any():
                    print(
                        f"Column {i} contains non-positive values."
                        f" Adding a small const. {self.dataframe[i].unique()}"
                    )
                    min_value = self.dataframe[i].min()
                    self.dataframe[i] += -min_value + 1e-6

                # Visualize data distribution
                # sns.histplot(self.dataframe[i], kde=True)
                # plt.title(f"Distribution of column {i}")
                # plt.show()

                # Calculate the optimal lambda for the Box-Cox transformation
                lambda_optimal = boxcox_normmax(self.dataframe[i] + 1)

                # Apply the Box-Cox transformation with the optimal lambda
                self.dataframe[i] = boxcox1p(self.dataframe[i], lambda_optimal)
                self.__skew_params[i] = {
                    "method": "boxcox",
                    "lambda": lambda_optimal,
                }

            except Exception as e:
                print(f"Box-Cox transformation failed for column {i}: {e}")
                print("Trying Yeo-Johnson transformation.")

                # Use Yeo-Johnson transformation as a fallback
                pt = PowerTransformer(method="yeo-johnson")
                self.dataframe[i] = pt.fit_transform(
                    self.dataframe[i].values.reshape(-1, 1)
                )
                self.__skew_params[i] = {"method": "yeo-johnson", "params": pt}
        with open(
            os.path.join("app/saved-models/skew/trans_params.sav"),
            "wb",
        ) as file:
            joblib.dump(self.__skew_params, file)
