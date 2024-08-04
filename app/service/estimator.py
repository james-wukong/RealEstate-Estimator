import datetime
import os
from typing import TypeVar, Union
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


from app.model.property import AddressReq, PropertyBase
from app.model.estimator import EstimatePriceModel
from app.service.property import attom_property_api
from app.api.utilities import init_dataframe
from app.api.constant import GrpColumns
from app.service.processor import Processor

ModelT = TypeVar(
    "ModelT",
    bound=Union[
        ElasticNetCV,
        LassoCV,
        RidgeCV,
        SVR,
        StackingCVRegressor,
        XGBRegressor,
        LGBMRegressor,
        GradientBoostingRegressor,
    ],
)


def load_model(
    filename: str,
    v: ModelT,
    basepath: str = "app/saved-models",
) -> ModelT:
    f = os.path.join(basepath, filename)
    model: v = joblib.load(f)

    return model


async def get_property_est(address: AddressReq) -> EstimatePriceModel:
    resp = await attom_property_api(
        address=address,
        identifier="/property/expandedprofile",
    )
    prop_data = resp.property[0]
    df: pd.DataFrame = process_data(load_data(api_data=prop_data))
    loaded_model_en = load_model("elastic_model.sav", ElasticNetCV)
    loaded_model_la = load_model("lasso_model.sav", LassoCV)
    loaded_model_ri = load_model("ridge_model.sav", RidgeCV)
    loaded_model_svr = load_model("svr_model.sav", SVR)
    loaded_model_gb = load_model("gbr_model.sav", GradientBoostingRegressor)
    loaded_model_lgb = load_model("lgb_model.sav", LGBMRegressor)
    loaded_model_xgb = load_model("xgb_model.sav", XGBRegressor)
    loaded_model_gen = load_model("stack_gen_model.sav", StackingCVRegressor)
    # yhat = loaded_model.predict(data)
    if "list_price" in df.columns:
        df.drop(columns=["list_price"], axis=1, inplace=True)
    yhat = (
        (0.04 * loaded_model_en.predict(df))
        + (0.04 * loaded_model_la.predict(df))
        + (0.04 * loaded_model_ri.predict(df))
        + (0.04 * loaded_model_svr.predict(df))
        + (0.13 * loaded_model_gb.predict(df))
        + (0.23 * loaded_model_lgb.predict(df))
        + (0.13 * loaded_model_xgb.predict(df))
        + (0.35 * loaded_model_gen.predict(np.array(df)))
    )
    # yhat = loaded_model_xgb.predict(df)
    # yhat = loaded_model_la.predict(df)
    # reverse predicted value to original value
    yhat = np.expm1(yhat)
    if prop_data.sale.amount.sale_amt != 0:
        sale_amt = prop_data.sale.amount.sale_amt
    else:
        sale_amt = prop_data.assessment.assessed.assd_ttl_value

    return EstimatePriceModel(
        est_price=round(yhat[0], 2),
        assd_price=prop_data.assessment.assessed.assd_ttl_value,
        mkt_price=prop_data.assessment.market.mkt_ttl_value,
        sale_price=sale_amt,
    )


def load_data(api_data: PropertyBase) -> pd.DataFrame:
    df = init_dataframe(filename="dataframe_tpl.csv")
    df.loc[0, "above_grade_finished_area"] = api_data.building.size.living_size
    # df.appliances = ""
    # df.architectural_style = ""
    df.loc[0, "year_built"] = int(api_data.summary.year_built)
    df.loc[0, "association_yn"] = True
    if api_data.building.interior.bsmt_size > 0:
        df.loc[0, "basement_yn"] = True
    else:
        df.loc[0, "basement_yn"] = False
    if api_data.building.interior.fplc_type.lower() == "yes":
        df.loc[0, "fireplace_yn"] = True
    else:
        df.loc[0, "fireplace_yn"] = False
    # df.basement = ""
    df.loc[0, "bathrooms_full"] = api_data.building.rooms.baths_full
    # df.bathrooms_half = 0
    df.loc[0, "bathrooms_total_integer"] = api_data.building.rooms.baths_total
    df.loc[0, "bedrooms_total"] = api_data.building.rooms.rooms_total
    df.loc[0, "below_grade_finished_area"] = api_data.building.interior.bsmt_size
    df.loc[0, "building_area_total"] = api_data.building.size.bldg_size
    # df.car__acres_cleared = 0
    # df.car__admin_fee = 0
    # df.car__application_fee = 0
    df.loc[0, "car__assigned_spaces"] = 1 if api_data.building.parking else 0
    # df.car__association_annual_expense = 0
    # df.car__bedroom_basement = 0
    # df.car__bedroom_lower = 0
    df.loc[0, "car__down_payment_resource_yn"] = True
    df.loc[0, "car__room_count"] = (
        api_data.building.rooms.baths_total + api_data.building.rooms.rooms_total
    )
    df.loc[0, "car__sq_ft_main"] = api_data.building.size.living_size
    if api_data.sale.amount.sale_amt != 0:
        sale_amt = api_data.sale.amount.sale_amt
    else:
        sale_amt = api_data.assessment.assessed.assd_ttl_value

    df.loc[0, "car_ratio__current_price__by__total_property_hla"] = round(
        sale_amt / api_data.building.size.living_size, 2
    )
    df.loc[0, "city"] = api_data.address.locality
    df.loc[0, "garage_spaces"] = api_data.building.parking.prkg_space
    df.loc[0, "garage_yn"] = True if api_data.building.parking.garage_type else False
    df.loc[0, "garage_spaces"] = api_data.building.parking.prkg_space
    df.loc[0, "heating"] = api_data.utilities.heating_type.title()
    df.loc[0, "latitude"] = float(api_data.location.latitude)
    df.loc[0, "longitude"] = float(api_data.location.longitude)
    # df.level = api_data.building.summary.levels
    df.loc[0, "levels"] = "{Two}"
    df.loc[0, "listing_contract_date"] = api_data.sale.amount.sale_rec_date
    df.loc[0, "original_entry_timestamp"] = datetime.date.today()
    df.loc[0, "original_list_price"] = sale_amt
    df.loc[0, "living_area"] = api_data.building.size.living_size
    df.loc[0, "street_name"] = api_data.address.situs_street_name
    df.loc[0, "tax_assessed_value"] = api_data.assessment.market.mkt_ttl_value
    df.loc[0, "property_sub_type"] = api_data.summary.prop_sub_type.title()
    df.loc[0, "property_type"] = api_data.summary.prop_type.title()
    df.loc[0, "lot_size_area"] = api_data.lot.lot_size1

    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    proc = Processor(df=df)
    proc.remove_cols(cols=GrpColumns.ZERO_COLS).update_binary_cols(
        cols=GrpColumns.BOOL_COLS
    ).update_int_cols(cols=GrpColumns.INT_COLS).update_num_cols(
        cols=GrpColumns.FLOAT_COLS
    ).update_str_cols(
        cols=GrpColumns.STR_COLS
    ).mlb_transform_cols(
        cols=GrpColumns.ENGINEER_COLS
    ).extract_date(
        cols=GrpColumns.DATE_COLS
    ).apply_skew_trans().remove_cols(
        cols=GrpColumns.IRRELEVANT_COLS
    ).apply_one_hot_enc(
        cols=GrpColumns.OH_ENC
    ).apply_label_enc(
        cols=GrpColumns.LABEL_ENC
    ).apply_target_enc(
        cols=GrpColumns.TARGET_ENC
    ).remove_cols(
        cols=GrpColumns.CORR_COLS
    ).remove_cols(
        cols=GrpColumns.REMOVE_COLS
    ).sort_features()

    return proc.df


def rmsle(y: float, y_pred: float) -> float:
    return np.sqrt(mean_squared_error(y, y_pred))
