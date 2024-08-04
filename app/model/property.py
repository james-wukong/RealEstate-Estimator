import datetime
from enum import Enum, unique
from pydantic import BaseModel, Field

from app.model.base import SearchBase, ConfigModel


@unique
class PropertyType(str, Enum):
    ARGI: str = "AGRICULTURAL (NEC)"
    APAR: str = "APARTMENT"
    CABI: str = "CABIN"
    CLUB: str = "CLUB"
    COMA: str = "COMMON AREA"
    COMB: str = "COMMERCIAL BUILDING"
    COMC: str = "COMMERCIAL CONDOMINIUM"
    COMN: str = "COMMERCIAL (NEC)"
    COND: str = "CONDOMINIUM"
    CONR: str = "CONVERTED RESIDENCE"
    DUPL: str = "DUPLEX"
    FARM: str = "FARMS"
    FFFR: str = "FAST FOOD FRANCHISE"
    FEDP: str = "FEDERAL PROPERTY"
    FORE: str = "FOREST"
    GRPQ: str = "GROUP QUARTERS"
    INDN: str = "INDUSTRIAL (NEC)"
    INDP: str = "INDUSTRIAL PLANT"
    MANH: str = "MANUFACTURED HOME"
    MARF: str = "MARINA FACILITY"
    MISC: str = "MISCELLANEOUS"
    MOBH: str = "MOBILE HOME"
    MOHL: str = "MOBILE HOME LOT"
    MOHP: str = "MOBILE HOME PARK"
    MUFD: str = "MULTI FAMILY DWELLING"
    NURH: str = "NURSERY/HORTICULTURE"
    OFFB: str = "OFFICE BUILDING"
    PUBN: str = "PUBLIC (NEC)"
    RELI: str = "RELIGIOUS"
    RESA: str = "RESIDENTIAL ACREAGE"
    RESL: str = "RESIDENTIAL LOT"
    RESN: str = "RESIDENTIAL (NEC)"
    RETT: str = "RETAIL TRADE"
    SFR: str = "SFR"
    STAP: str = "STATE PROPERTY"
    STOO: str = "STORES & OFFICES"
    STOR: str = "STORES & RESIDENTIAL"
    TAXE: str = "TAX EXEMPT"
    TOWN: str = "TOWNHOUSE/ROWHOUSE"
    TRIP: str = "TRIPLEX"
    UTIL: str = "UTILITIES"
    VALN: str = "VACANT LAND (NEC)"


# Properties to receive on sending api request
class AddressReq(SearchBase):
    postalcode: str = Field(
        default="",
        title="The zip code or postal code to search",
        max_length=25,
    )
    propertytype: PropertyType | None = Field(
        default=None,
        title="A specific property classification such as 'appartment'",
        max_length=100,
    )
    address1: str = Field(
        default="",
        title="The first line of the property address",
        max_length=100,
    )
    address2: str = Field(
        default="",
        title="The second line of the property address",
        max_length=100,
    )


class AddressModel(ConfigModel):
    country: str = ""
    country_subd: str = Field("", alias="countrySubd")
    line1: str = ""
    line2: str = ""
    locality: str = ""
    one_line: str = Field("", alias="oneLine")
    postal1: str = ""
    postal2: str = ""
    postal3: str = ""
    situs_house_num: str = Field(
        default="",
        max_length=10,
        alias="situsHouseNumber",
    )
    situs_street_name: str = Field(
        default="",
        max_length=100,
        alias="situsStreetName",
    )
    situs_address_sfx: str = Field(
        default="",
        max_length=10,
        alias="situsAddressSuffix",
    )


class LocationModel(ConfigModel):
    accuracy: str = ""
    latitude: str = ""
    longitude: str = ""
    distance: float = 0
    geoid: str = ""
    geoid_v4: dict[str, str] | None = Field(None, alias="geoIdV4")


class IdentifierModel(ConfigModel):
    id: int = Field(0, alias="Id")
    fips: str = ""
    apn: str = ""
    attom_id: int = Field(0, alias="attomId")


class LotModel(ConfigModel):
    lot_num: str = Field("", alias="lotNum")
    lot_size1: float = Field(0, alias="lotSize1")
    lot_size2: int = Field(0, alias="lotSize2")
    zoning_type: str = Field("", alias="zoningType")
    site_zoning_ident: str = Field("", alias="siteZoningIdent")
    pool_type: str = Field("", alias="poolType")


class AssessedModel(ConfigModel):
    assd_impr_value: int = Field(0, alias="assdImprValue")
    assd_land_value: int = Field(0, alias="assdLandValue")
    assd_ttl_value: int = Field(0, alias="assdTtlValue")


class MarketModel(ConfigModel):
    mkt_impr_value: int = Field(0, alias="mktImprValue")
    mkt_land_value: int = Field(0, alias="mktLandValue")
    mkt_ttl_value: int = Field(0, alias="mktTtlValue")


class TaxModel(ConfigModel):
    tax_amt: float = Field(0, alias="taxAmt")
    tax_per_size_unit: float = Field(0, alias="taxPerSizeUnit")
    tax_year: float = Field(0, alias="taxYear")
    # exemption: float | None = None
    # exemptiontype: float | None = None


class AssessmentModel(ConfigModel):
    improvement_percent: int | float = Field(
        0,
        alias="improvementPercent",
    )
    assessed: AssessedModel | None = None
    market: MarketModel | None = None
    tax: TaxModel | None = None


class UtilitiesModel(ConfigModel):
    heating_feul: str = Field("", alias="heatingFuel")
    heating_type: str = Field("", alias="heatingType")
    cooling_type: str = Field("", alias="coolingType")


class SaleAmountModel(ConfigModel):
    sale_amt: float = Field(0, alias="saleAmt")
    sale_rec_date: datetime.date | None = Field(None, alias="saleRecDate")
    sale_disclosure_type: int = Field(
        0,
        alias="saleDisclosureType",
    )
    sale_doc_type: str = Field(0, alias="saleDocType")
    sale_doc_num: str = Field(0, alias="saleDocNum")
    sale_trans_type: str = Field(0, alias="saleTransType")


class CalculationModel(ConfigModel):
    price_per_bed: float = Field(0, alias="pricePerBed")
    # sales amount / living size
    price_per_size_unit: float = Field(0, alias="pricePerSizeUnit")


class SaleModel(ConfigModel):
    sale_search_date: datetime.date | None = Field(
        None,
        alias="saleSearchDate",
    )
    sale_trans_date: datetime.date | None = Field(None, alias="saleTransDate")
    transaction_ident: str | None = Field(None, alias="transactionIdent")
    sale_amount_data: SaleAmountModel | None = Field(
        None,
        alias="saleAmountData",
    )
    amount: SaleAmountModel | None = None
    calculation: CalculationModel | None = None


class Summary(ConfigModel):
    arch_style: str = Field("", alias="archStyle")
    absentee_ind: str = Field("", alias="absenteeInd")
    prop_class: str = Field("", alias="propClass")
    prop_sub_type: str = Field("", alias="propSubType")
    prop_type: str = Field("", alias="propType")
    property_type: str = Field("", alias="propertyType")
    year_built: int = Field("", alias="yearBuilt")
    prop_land_use: str = Field("", alias="propLandUse")
    prop_indicator: int = Field("", alias="propIndicator")
    legal1: str = Field("", alias="legal1")
    quit_claim_flag: str = Field("", alias="quitClaimFlag")
    reo_flag: str = Field("", alias="REOflag")


class SizeModel(ConfigModel):
    bldg_size: float = Field(0, alias="bldgSize")
    gross_size: float = Field(0, alias="grossSize")
    gross_size_adjusted: float = Field(0, alias="grossSizeAdjusted")
    ground_floor_size: float = Field(0, alias="groundFloorSize")
    living_size: float = Field(0, alias="livingSize")
    groununiversal_sized_floor_size: float = Field(0, alias="universalSize")
    size_ind: str = Field("", alias="sizeInd")


class RoomsModel(ConfigModel):
    bath_fixtures: int = Field(0, alias="bathFixtures")
    baths_full: int = Field(0, alias="bathsFull")
    baths_total: int = Field(0, alias="bathsTotal")
    beds: int = 0
    rooms_total: int = Field(0, alias="roomsTotal")


class InteriorModel(ConfigModel):
    bsmt_size: int = Field(0, alias="bsmtSize")
    bsmt_finished_percent: float = Field(0, alias="bsmtFinishedPercent")
    fplc_count: int = Field(0, alias="fplcCount")
    fplc_ind: str = Field("", alias="fplcInd")
    fplc_type: str = Field("", alias="fplcType")


class ConstructionModel(ConfigModel):
    condition: str = ""
    wall_type: str = Field("", alias="wallType")
    property_structure_major_improvements_year: str = Field(
        "", alias="propertyStructureMajorImprovementsYear"
    )


class ParkingModel(ConfigModel):
    garage_type: str = Field("", alias="garageType")
    prkg_size: float = Field(0, alias="prkgSize")
    prkg_type: str = Field("", alias="prkgType")
    prkg_space: str = Field("0", alias="prkgSpaces")


class RSummaryModel(ConfigModel):
    levels: int = 0
    units_count: int = Field(0, alias="unitsCount")
    view: str = ""
    view_code: str = Field("", alias="viewCode")


class BuildingModel(ConfigModel):
    size: SizeModel | None = None
    rooms: RoomsModel | None = None
    interior: InteriorModel | None = None
    construction: ConstructionModel | None = None
    parking: ParkingModel | None = None
    summary: RSummaryModel | None = None


class StatusModel(ConfigModel):
    version: str = ""
    code: int
    msg: str = ""
    total: int
    page: int
    pagesize: int
    transaction_id: str = Field("", alias="transactionID")


# Properties to receive on getting api response
class PropertyBase(ConfigModel):
    identifier: IdentifierModel
    lot: LotModel | None = None
    address: AddressModel
    location: LocationModel
    sale: SaleModel | None = None
    assessment: AssessmentModel | None = None
    building: BuildingModel | None = None
    utilities: UtilitiesModel | None = None
    summary: Summary | None = None


class PropertyListResp(ConfigModel):
    status: StatusModel
    property: list[PropertyBase]


class PropertyResp(BaseModel):
    total: int = 0
    postal: str | None = None
    avg_mkt_value: float = Field(
        default=0,
        ge=0,
        title="average market price",
        examples=[1999.0],
    )
    avg_assd_value: float = Field(
        default=0,
        ge=0,
        title="average assessed price",
        examples=[1999.0],
    )
    ttl_mkt_value: float = Field(
        default=0,
        ge=0,
        title="total market price",
        examples=[1999.0],
    )
    ttl_assd_value: float = Field(
        default=0,
        ge=0,
        title="total assessed price",
        examples=[1999.0],
    )
    avg_unit_price: float = Field(
        default=0,
        ge=0,
        title="total assessed price",
        examples=[1999.0],
    )
