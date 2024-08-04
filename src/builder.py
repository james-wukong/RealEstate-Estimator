from src.clean import DataClean
from src.constants import GrpColumns

# from sklearn.preprocessing import StandardScaler
# import numpy as np


# from scipy.special import boxcox1p
# from scipy.stats import boxcox_normmax
# from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.svm import SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
# from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import mean_squared_error
# from mlxtend.regressor import StackingCVRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor


class Builder:

    def __init__(self, filename: str) -> None:
        self.dc = DataClean(file=filename)

    def build_data(self) -> None:
        """_summary_"""
        self.dc.replace_empty_brackets()
        self.dc.drop_dirty_rows()
        # update association yes/no
        association_fields = [
            "association_fee_frequency",
            "association_fee",
            "association_fee2_frequency",
            "association_fee2",
            "association_name",
            "association_name2",
            "association_phone",
            "association_phone2",
        ]
        self.dc.update_association_yn(
            fields=association_fields,
            target_col="association_yn",
        )
        # update basement_yn
        self.dc.update_dependant_yn(depends="basement", target="basement_yn")
        # update fireplace_yn
        self.dc.update_dependant_yn(
            depends="fireplace_features",
            target="fireplace_yn",
        )
        # update car__special_assessment_yn
        self.dc.update_dependant_yn(
            depends="car__special_assessment_description",
            target="car__special_assessment_yn",
        )
        self.dc.update_binary_cols(cols=GrpColumns.BOOL_COLS)
        self.dc.update_num_cols(cols=GrpColumns.FLOAT_COLS)
        self.dc.update_int_cols(cols=GrpColumns.INT_COLS)
        self.dc.update_str_cols(cols=GrpColumns.STR_COLS)
        self.dc.mlb_transform_cols(cols=GrpColumns.ENGINEER_COLS)
        self.dc.extract_date(cols=GrpColumns.DATE_COLS)
        self.dc.update_skewed()
        self.dc.drop_cols(
            cols=list(
                set(GrpColumns.IRRELEVANT_COLS).union(
                    set(GrpColumns.REMOVE_COLS),
                )
                - set(self.dc.removed_cols),
            )
        )
        self.dc.remove_missing(100)
        self.dc.one_hot_enc(GrpColumns.OH_ENC)
        self.dc.label_enc(GrpColumns.LABEL_ENC)
        self.dc.target_enc(GrpColumns.TARGET_ENC)
