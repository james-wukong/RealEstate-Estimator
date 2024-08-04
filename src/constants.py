from enum import Enum


class FreqConst(int, Enum):
    MONTHLY = 12
    QUARTERLY = 4
    SEMI_ANNUALLY = 2
    ANNUALLY = 1
    NAN = 0


class GrpColumns:
    """CONSTANT feature is based on the raw data"""

    Y_COL = "list_price"

    DATE_COLS = [
        # 0 missing value
        "listing_contract_date",  # 1
        "original_entry_timestamp",
    ]

    # features that are supposed to be boolean type
    BOOL_COLS = {
        # false fillin
        "false": [
            "car__land_included_yn",  # 1
            "car__main_level_garage_yn",  # 1
            "car__down_payment_resource_yn",
            "carport_yn",  # 1
            "garage_yn",
        ],
        # most frequent fillin
        "freq": [
            "new_construction_yn",  # 1
            "open_parking_yn",  # 1
            "senior_community_yn",  # 1
        ],
        # other
        "other": [
            # 0 missing values
            "fireplace_yn",  # 1
            "association_yn",
            "basement_yn",
            # removed
        ],
    }

    # features that of type float
    FLOAT_COLS = {
        "median": [
            # fillin missing with median value
            "above_grade_finished_area",  # 1
            "below_grade_finished_area",  # 1
            "living_area",  # 1
            "lot_size_area",  # 1
            "original_list_price",  # 1
            "tax_assessed_value",  # 1
        ],
        "zero": [
            # fillin missing with 0 value
            "car__association_annual_expense",  # 1
            "car__acres_cleared",  # 1
            "car__admin_fee",  # 1
            "car__application_fee",  # 1
            "car__sq_ft_unheated_basement",  # 1
            "car__sq_ft_unheated_lower",  # 1
            "car__sq_ft_unheated_main",  # 1
            "car__sq_ft_unheated_third",  # 1
            "car__sq_ft_unheated_total",  # 1
            "car__sq_ft_unheated_upper",  # 1
            "car__sq_ft_garage",  # 1
            "car__sq_ft_lower",  # 1
            "car__sq_ft_main",  # 1
            "car__sq_ft_third",  # 1
            "car__sq_ft_total_property_hla",  # 1
            "car__sq_ft_upper",  # 1
            "car_ratio__current_price__by__total_property_hla",  # 1
            # median if there is association, else 0
        ],
        "other": [
            # no missing values
            "latitude",  # 1
            "longitude",  # 1
            # "list_price",  # 1
            # removed
        ],
    }

    # features that of type integer
    INT_COLS = {
        "median": [
            # fillin missing with median value
        ],
        "zero": [
            # fillin missing with 0 value
            "car__bedroom_basement",  # 1
            "car__bedroom_lower",  # 1
            "car__bedroom_main",  # 1
            "car__bedroom_second_lq",  # 1
            "car__bedroom_third",  # 1
            "car__bedroom_upper",  # 1
            "car__green_verification_count",  # 1
            "car__full_bath_basement",  # 1
            "car__full_bath_lower",  # 1
            "car__full_bath_main",  # 1
            "car__full_bath_second_lq",  # 1
            "car__full_bath_third",  # 1
            "car__full_bath_upper",  # 1
            "car__half_bath_basement",  # 1
            "car__half_bath_lower",  # 1
            "car__half_bath_main",  # 1
            "car__half_bath_second_lq",  # 1
            "car__half_bath_third",  # 1
            "car__half_bath_upper",  # 1
            # few data available so fill in with 0
            "car__number_of_bays",  # 1
            "car__number_of_bedrooms_septic",  # 1
            "car__number_of_docks_total",  # 1
            "car__number_of_drive_in_doors_total",  # 1
            "car__security_deposit",  # 1
            "carport_spaces",  # 1
            "car__assigned_spaces",  # 1
            "car__room_count",  # 1
            "covered_spaces",  # 1
            "photos_count",
        ],
        "freq": [
            # fillin missing with most frequently used value
            "year_built",  # 1
            "bathrooms_full",  # 1
            "bathrooms_half",  # 1
            "bathrooms_total_integer",  # 1
            "bedrooms_total",  # 1
            "documents_count",  # 1
        ],
        "other": [
            # no missing values
            # "documents_count",  # 1
        ],
    }

    # features that of type string
    STR_COLS = {
        "freq": [
            # fill in with most frequently used value (7% missing)
            # Percentage: 79.91
            "car__entry_location_mls",
            # 44% missing
            # 94.21% missing
            "car__zoning_specification",
            "county_or_parish",
            "levels",  # 1
            "state_or_province",  # 1
            "property_type",  # 1
            "property_sub_type",  # 1
            "street_name",  # 1
        ],
        "Unspecified": [
            # few missing, fill in with "Unspecified"
            "elementary_school",  # 1
            # few missing, fill in with "Unspecified"
            "high_school",  # 1
            # few missing, fill in with "Unspecified"
            "middle_or_junior_school",  # 1
        ],
        "None": [
            "car_hoa_subject_to",  # 1"
        ],
        "other": [
            # no missing
            "city",  # 1
            "postal_code",  # 1
        ],
    }

    # multilabel encoding cols
    ENGINEER_COLS = [
        "accessibility_features",  # 1
        "appliances",  # 1
        "architectural_style",  # 1
        # 14% missing
        "car__construction_type",  # 1
        "construction_materials",  # 1
        "cooling",  # 1
        "door_features",  # 1
        "exterior_features",  # 1
        "fencing",  # 1
        "flooring",  # 1
        "foundation_details",  # 1
        "heating",  # 1
        "interior_features",  # 1
        "laundry_features",  # 1
        "lot_features",  # 1
        "parking_features",  # 1
        "road_responsibility",  # 1
        "road_surface_type",  # 1
        "roof",  # 1
        "security_features",  # 1
        "building_features",  # 1
        "sewer",  # 1
        "utilities",  # 1
        "view",  # 1
        "window_features",  # 1
    ]

    TARGET_ENC = [
        "city",
        "postal_code",
        "elementary_school",
        "high_school",
        "middle_or_junior_school",
        "street_name",
        "car__zoning_specification",
        "county_or_parish",
        "property_sub_type",
    ]

    OH_ENC = [
        "state_or_province",
        "property_type",
        "car_hoa_subject_to",
    ]

    LABEL_ENC = [
        "car__entry_location_mls",
        "levels",
    ]

    REMOVE_COLS = [
        "building_features",
        "street_number",
    ]

    # irrelevant features that need to remove
    IRRELEVANT_COLS = [
        "id",
        "additional_parcels_description",
        "association_fee_frequency",
        "association_fee2_frequency",
        "association_fee",
        "association_fee2",
        "association_name",
        "association_name2",
        "association_phone",
        "association_phone2",
        "availability_date",
        "basement",
        "builder_model",
        "builder_name",
        "building_area_total",
        "buyer_agency_compensation_type",
        "buyer_agency_compensation",
        "buyer_agent_aor",
        "buyer_agent_direct_phone",
        "buyer_agent_email",
        "buyer_agent_full_name",
        "buyer_agent_key",
        "buyer_agent_mls_id",
        "buyer_financing",
        "buyer_office_key",
        "buyer_office_mls_id",
        "buyer_office_name",
        "buyer_office_phone",
        "buyer_team_key",
        "buyer_team_name",
        "car__association_email",
        "car__association_email2",
        "car__assumable",
        "car__attribution_contact",
        "car__attribution_name",
        "car__attribution_type_listing",
        "car__auction_bid_information",
        "car__auction_bid_type",
        "car__auction_yn",  # 1
        "car__bonus_amount",  # 1
        "car__buyer_team_id",
        "car__buyer_team_mls_id",
        "car__can_subdivide_yn",  # 1
        "car__ceiling_height_feet",
        "car__ceiling_height_inches",
        "car__city_taxes_paid_to",
        "car__closed_comp_type",
        "car__co_buyer_team_id",
        "car__co_buyer_team_key",
        "car__co_buyer_team_mls_id",
        "car__co_buyer_team_name",
        "car__co_list_team_id",
        "car__co_list_team_key",
        "car__co_list_team_mls_id",
        "car__co_list_team_name",
        "car__commercial_location_description",
        "car__comp_sale_yn",  # 1
        "car__compensation_remarks",
        "car__complex_name",
        "car__days_on_market_to_close",
        "car__deed_reference",
        "car__due_diligence_period_end_date",
        "car__due_diligence_yn",
        "car__easement",
        "car__exception_yn",
        "car__fire_sprinkler_type",
        "car__flood_plain",
        "car__foundation_details_proposed",
        "car__geocode_confidence",
        "car__hold_date",
        "car__inside_city_yn",
        "car__list_team_id",
        "car__list_team_mls_id",
        "car__mls_major_change_type",
        "car__move_in_fee",  # 1
        "car__number_of_completed_units_total",  # 1
        "car__number_of_projected_units_total",  # 1
        "car__owner_agent_yn",  # 1
        "car__ownership_period",  # 1
        "car__pet_deposit",  # 1
        "car__plat_book_slide",
        "car__plat_reference_section_pages",
        "car__power_production_ownership",
        "car__power_production_size",
        "car__power_production_year",
        "car__property_sub_type_additional",
        "car__proposed_completion_date",
        "car__proposed_special_assessment_description",
        "car__proposed_special_assessment_yn",
        "car__rail_service",
        "car__restrictions_description",
        "car__restrictions",
        "car__road_frontage",  # 1
        "car__room_other",
        "car__second_living_quarters",
        "car__special_assessment_description",
        "car__smoking_allowed_yn",  # 1
        "car__special_assessment_yn",
        "car__sq_ft_available_maximum",
        "car__sq_ft_available_minimum",
        "car__sq_ft_building_minimum",
        "car__sq_ft_maximum_lease",
        "car__sq_ft_minimum_lease",
        "car__sq_ft_office",
        "car__sq_ft_other",
        "car__sq_ft_second_living_quarters_hla",
        "car__sq_ft_second_living_quarters",
        "car__sq_ft_warehouse",  # 1
        "car__syndicate_participation",  # 1
        "car__transaction_type",
        "car__web_url",
        "car_ccr_subject_to",
        "car_hoa_subject_to_dues",  # 1
        "close_date",
        "close_price",
        "co_buyer_agent_aor",
        "co_buyer_agent_direct_phone",
        "co_buyer_agent_email",
        "co_buyer_agent_full_name",
        "co_buyer_agent_key",
        "co_buyer_agent_mls_id",
        "co_buyer_office_key",
        "co_buyer_office_mls_id",
        "co_buyer_office_name",
        "co_buyer_office_phone",
        "co_list_agent_aor",
        "co_list_agent_direct_phone",
        "co_list_agent_email",
        "co_list_agent_full_name",
        "co_list_agent_key",
        "co_list_agent_mls_id",
        "co_list_office_key",
        "co_list_office_mls_id",
        "co_list_office_name",
        "co_list_office_phone",
        "community_features",
        "concessions_amount",
        "concessions_comments",
        "cross_street",
        "cumulative_days_on_market",
        "days_on_market",
        "development_status",
        "directions",
        "documents_available",  # 1
        "documents_change_timestamp",
        "dual_variable_compensation_yn",
        "elevation",
        "entry_level",
        "entry_location",
        "exclusions",
        "fireplace_features",
        "furnished",  # 1
        "garage_spaces",  # 1
        "green_sustainability",
        "gross_scheduled_income",
        "habitable_residence_yn",
        "horse_amenities",  # 1
        "inclusions",
        "internet_address_display_yn",
        "internet_automated_valuation_display_yn",
        "internet_consumer_comment_yn",
        "internet_entire_listing_display_yn",
        "lease_considered_yn",
        "lease_term",
        "list_agent_aor",
        "list_agent_direct_phone",
        "list_agent_email",
        "list_agent_full_name",
        "list_agent_key",
        "list_agent_mls_id",
        "list_aor",
        "list_office_aor",
        "list_office_key",
        "list_office_mls_id",
        "list_office_name",
        "list_office_phone",
        "list_team_key",
        "list_team_name",
        "listing_agreement",
        # "listing_contract_date",
        "listing_key",
        "listing_service",
        "listing_terms",
        "living_area_units",
        "lot_size_acres",
        "lot_size_dimensions",
        "lot_size_square_feet",
        "lot_size_units",
        "major_change_timestamp",
        "major_change_type",
        "media",
        "mlg_can_use",
        "mlg_can_view",
        "mls_status",
        "mlsgrid_listing_id",
        "modification_timestamp",
        "net_operating_income",
        "number_of_buildings",
        "number_of_units_leased",
        "number_of_units_total",
        "odata_id",
        "off_market_date",
        "open_parking_spaces",
        "operating_expense",
        "originating_system_modification_timestamp",
        "originating_system_name",
        "other_equipment",
        "other_parking",
        "other_structures",
        "owner_name",
        "owner_pays",
        "owner_phone",
        "parcel_number",
        "parking_total",
        "patio_and_porch_features",
        "pets_allowed",
        "photos_change_timestamp",
        "possible_use",
        "postal_code_plus4",
        "previous_list_price",
        "price_change_timestamp",
        "private_office_remarks",
        "private_remarks",
        "property_attached_yn",
        "public_remarks",
        "purchase_contract_date",
        "road_frontage_type",  # 1
        "rooms",
        "showing_contact_phone",
        "showing_requirements",
        "special_listing_conditions",
        "standard_status",
        "status_change_timestamp",
        "stories",
        "street_dir_prefix",
        "street_dir_suffix",
        "street_number_numeric",
        "street_suffix",
        "sub_agency_compensation_type",
        "sub_agency_compensation",
        "subdivision_name",
        "syndicate_to",
        "syndication_remarks",
        "tax_block",
        "tax_legal_description",
        "tenant_pays",
        "transaction_broker_compensation_type",
        "transaction_broker_compensation",
        "unit_number",
        "unit_types",
        "virtual_tour_url_branded",
        "virtual_tour_url_unbranded",
        "water_body_name",
        "water_source",
        "waterfront_features",
        "withdrawn_date",
        "wooded_area",  # 1
        "zoning",
        "inserted_at",
        "updated_at",
        "listing_id",
    ]

    # columns that don't have any data
    ZERO_COLS = [
        "above_grade_finished_area_source",
        "above_grade_finished_area_units",
        "additional_parcels_yn",
        "below_grade_finished_area_source",
        "below_grade_finished_area_units",
        "building_area_source",
        "building_area_units",
        "buyer_brokerage_compensation_type",
        "buyer_brokerage_compensation",
        "buyer_office_phone_ext",
        "cancellation_date",
        "car__financing_information",
        "car__primary_residence_yn",
        "car__total_documents_count",
        "car__total_photos_count",
        "co_buyer_office_phone_ext",
        "co_list_office_phone_ext",
        "comp_sale_yn",
        "contingent_date",
        "country",
        "dual_or_variable_rate_commission_yn",
        "green_energy_generation",
        "gross_income",
        "list_office_phone_ext",
        "living_area_source",
        "pool_features",
        "showing_contact_type",
        "stories_total",
        "structure_type",
        "tax_year",
        "unknown_fields",
        "vacancy_allowance_rate",
        "year_built_source",
        "zoning_description",
        # tmp rows
    ]
