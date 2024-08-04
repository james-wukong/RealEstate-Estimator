import re
from typing import Self
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# from sklearn.model_selection import KFold
from category_encoders import TargetEncoder
from dateutil.relativedelta import relativedelta

# from scipy.stats import skew
from scipy.special import boxcox1p

# from scipy.stats import boxcox_normmax
from sklearn.preprocessing import PowerTransformer

# from app.api.constant import GrpColumns


class Processor:
    def __init__(self, df: pd.DataFrame = None) -> None:
        if df is None:
            raise ValueError("dataframe must be provided")
        self.df = df
        self.mlb_params = {}

    def update_binary_cols(
        self,
        cols: dict[str, list[str]],
    ) -> Self:
        """_summary_

        Args:
            cols (dict[str, list[str]]): _description_

        Returns:
            Self: _description_
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
                    self.df[c] = (
                        self.df[c]
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
                # median value for these cols is True
                for c in col:
                    self.df[c] = (
                        self.df[c]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map(mapping)
                        .fillna(True)
                        .astype(bool)
                    )
            elif key == "other":
                self.df[col] = self.df[col].astype(np.bool_)
        return self

    def update_int_cols(
        self,
        cols: dict[str, list[str]],
    ) -> Self:
        """_summary_

        Args:
            cols (dict[str, list[str]]): _description_

        Returns:
            Self: _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "median":
                    pass
                elif key == "zero":
                    self.df[values] = (
                        self.df[values]
                        .fillna(0)
                        .astype(
                            np.int32,
                        )
                    )
                elif key == "freq":
                    for col in values:
                        # Calculate the mode of the feature and fill NaN values
                        match col:
                            case "year_built":
                                self.df[col] = self.df[col].replace(0, np.nan)
                                self.df[col] = (
                                    self.df[col].fillna(2024).astype(np.int32)
                                )
                            case "bathrooms_full":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(2)
                                    .astype(
                                        np.int32,
                                    )
                                )
                            case "bathrooms_total_integer" | "bedrooms_total":
                                # case "bedrooms_total":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(3)
                                    .astype(
                                        np.int32,
                                    )
                                )
                            case "bathrooms_half":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(0)
                                    .astype(
                                        np.int32,
                                    )
                                )
                elif key == "other":
                    self.df[values] = (
                        self.df[values]
                        .fillna(0)
                        .astype(
                            np.int32,
                        )
                    )
        return self

    def update_num_cols(
        self,
        cols: dict[str, list[str]],
    ) -> Self:
        """_summary_

        Args:
            cols (dict[str, list[str]]): _description_

        Returns:
            Self: _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "median":
                    for col in values:
                        # Convert to numeric and coerce errors to NaN
                        self.df[col] = pd.to_numeric(
                            self.df[col],
                            errors="coerce",
                        )
                        # Fill NaN values with median
                        match col:
                            case "above_grade_finished_area":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(1840.0)
                                    .astype(
                                        np.float32,
                                    )
                                )
                            case "below_grade_finished_area":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(0)
                                    .astype(
                                        np.float32,
                                    )
                                )
                            case "living_area":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(1932.0)
                                    .astype(
                                        np.float32,
                                    )
                                )
                            case "lot_size_area":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(0.31)
                                    .astype(
                                        np.float32,
                                    )
                                )
                            case "original_list_price":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(
                                        410000.0,
                                    )
                                    .astype(np.float32)
                                )
                            case "tax_assessed_value":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(
                                        285600.0,
                                    )
                                    .astype(np.float32)
                                )
                elif key == "zero":
                    for col in values:
                        if col in self.df.columns:
                            self.df[col] = (
                                self.df[col]
                                .fillna(0)
                                .astype(
                                    np.float32,
                                )
                            )
                elif key == "other":
                    self.df[values] = (
                        self.df[values]
                        .fillna(0.0)
                        .astype(
                            np.float32,
                        )
                    )
        return self

    def update_str_cols(
        self,
        cols: dict[str, list[str]],
    ) -> Self:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            cols (dict[str, list[str]]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        if cols:
            for key, values in cols.items():
                if key == "freq":
                    for col in values:
                        # Calculate the mode of the feature and fill NaN values
                        match col:
                            case "car__entry_location_mls":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("Main")
                                    .astype(
                                        str,
                                    )
                                )
                            case "car__zoning_specification":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("RES")
                                    .astype(
                                        str,
                                    )
                                )
                            case "county_or_parish":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("Mecklenburg")
                                    .astype(
                                        str,
                                    )
                                )
                            case "levels":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("{Two}")
                                    .astype(
                                        str,
                                    )
                                )
                            case "state_or_province":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("NC")
                                    .astype(
                                        str,
                                    )
                                )
                            case "property_type":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("Residential")
                                    .astype(
                                        str,
                                    )
                                )
                            case "property_sub_type":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna(
                                        "Single Family Residence",
                                    )
                                    .astype(str)
                                )
                            case "street_name":
                                self.df[col] = (
                                    self.df[col]
                                    .fillna("Deer Brook")
                                    .astype(
                                        str,
                                    )
                                )
                elif key in ["Unspecified", "None"]:
                    self.df[values] = self.df[values].fillna(key).astype(str)
                elif key == "other":
                    self.df[values] = (
                        self.df[values]
                        .fillna("")
                        .astype(
                            str,
                        )
                    )
        return self

    def mlb_transform_cols(
        self,
        cols: list[str],
    ) -> Self:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            cols (list[str]): _description_

        Returns:
            pd.DataFrame: _description_
        """

        def parse_categorical_cols(value: str) -> list[str]:
            if isinstance(value, float):
                value = str(value)

            value = re.sub(r"[^a-zA-Z0-9,-\/]", "", value)
            return value.split(",")

        # Function to apply MultiLabelBinarizer to a column
        def mlb_transform_column(
            mlb: MultiLabelBinarizer,
            col: str,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                mlb.fit_transform(self.df[col]),
                columns=mlb.classes_,
                index=self.df[col].index,
            )

        for col in cols:
            # load saved params
            with open(f"app/saved-models/mlb/{col}_mlb.sav", "rb") as file:
                loaded_mlb: MultiLabelBinarizer = joblib.load(file)
            self.df[col] = self.df[col].map(parse_categorical_cols)
            # Apply MultiLabelBinarizer to each column
            enc_col = pd.DataFrame(
                loaded_mlb.transform(self.df[col]),
                columns=[f"{col.lower()}_{c.lower()}" for c in loaded_mlb.classes_],
                index=self.df.index,
            ).astype(np.bool_)

            self.df = pd.concat([self.df, enc_col], axis=1)
            # drop the empty column
            if self.df.get(f"{col}_") is not None:
                # self.drop_cols([f"{col}_"])
                self.df.drop(columns=[f"{col}_"], inplace=True)
        # drop original cols
        self.df.drop(columns=cols, axis=1, inplace=True)

        return self

    def extract_date(
        self,
        cols: list[str],
    ) -> Self:
        def months_between(start_date: str, end_date: str) -> int:
            delta = relativedelta(end_date, start_date)
            return delta.years * 12 + delta.months

        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col])
            if col == "listing_contract_date":
                self.df[col] = self.df[col].fillna("2024-05-13 00:00:00")
            elif col == "original_entry_timestamp":
                self.df[col] = self.df[col].fillna("2024-05-12 03:48:00")
            self.df[col + "_year"] = self.df[col].dt.year
            self.df[col + "_month"] = self.df[col].dt.month
            self.df[col + "_day"] = self.df[col].dt.day
        self.df["months_between_list"] = self.df.apply(
            lambda row: months_between(
                row["original_entry_timestamp"], row["listing_contract_date"]
            ),
            axis=1,
        )
        today = pd.Timestamp("today")
        self.df["months_till_today"] = self.df["original_entry_timestamp"].apply(
            lambda start_date: months_between(start_date, today),
            # axis=1,
        )

        self.df.drop(columns=cols, axis=1, inplace=True)

        return self

    def apply_skew_trans(self) -> Self:
        # load saved params
        with open("app/saved-models/skew/trans_params.sav", "rb") as file:
            loaded_trans_params: dict = joblib.load(file)
        for feature, params in loaded_trans_params.items():
            if feature in self.df.columns:
                if params["method"] == "boxcox":
                    self.df[feature] = boxcox1p(
                        self.df[feature],
                        params["lambda"],
                    )
                elif params["method"] == "yeo-johnson":
                    pt: PowerTransformer = params["params"]
                    self.df[feature] = pt.transform(
                        self.df[feature].values.reshape(-1, 1)
                    )
        return self

    def remove_cols(self, cols: list[str]) -> Self:
        if set(cols).issubset(set(self.df.columns)):
            self.df.drop(columns=list(set(cols)), axis=1, inplace=True)
        else:
            rm_cols = set(cols) & set(self.df.columns)
            self.df.drop(columns=list(rm_cols), axis=1, inplace=True)

        return self

    def apply_one_hot_enc(self, cols: list[str]) -> Self:
        # Load the saved columns
        with open("app/saved-models/enc/one_hot_enc.sav", "rb") as file:
            encoded_columns = joblib.load(file)
        # print(encoded_columns)
        for col in cols:
            # if col in encoded_columns:
            dummies = pd.get_dummies(
                self.df[col],
                drop_first=True,
                prefix=col,
            )
            # Reindex to match training columns
            dummies = dummies.reindex(
                columns=[c for c in encoded_columns if c.startswith(f"{col}_")],
                fill_value=0,
            )

            self.df = pd.concat(
                [self.df, dummies],
                axis=1,
            )

        self.df.drop(cols, axis=1, inplace=True)

        return self

    def apply_label_enc(self, cols: list[str]) -> Self:
        # Load the saved encoders
        with open("app/saved-models/enc/label_enc.sav", "rb") as file:
            label_encoders = joblib.load(file)
        for col in cols:
            label_encoder: LabelEncoder = label_encoders[col]
            self.df[col + "_enc"] = label_encoder.transform(self.df[col])
        self.df.drop(cols, axis=1, inplace=True)

        return self

    def apply_target_enc(self, cols: list[str]) -> Self:
        for col in cols:
            # load encoder
            with open(
                "app/saved-models/enc/" + col + "_enc.sav",
                "rb",
            ) as file:
                loaded_encoder: TargetEncoder = joblib.load(file)
            self.df[f"{col}_enc"] = loaded_encoder.transform(
                self.df[[col]],
            )[col]

        self.df.drop(columns=cols, axis=1, inplace=True)
        return self

    def sort_features(self) -> Self:
        self.df = self.df.reindex(sorted(self.df.columns), axis=1)
        return self
