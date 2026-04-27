import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta


# House-keeping tables for Pandas data import

# Origination dataset specs
orig_cols = ["credit_score", "date_first_payment", "first_time_homebuyer", "date_maturity",
            "MSA", "insurance_percent", "nr_units", "occupancy", "original_combined_loan_to_value", 
            "original_debt_to_income", "original_upb", "original_loan_to_value", "original_interest_rate",
            "channel", "prepayment_penalty", "amortization_type", "property_state",
            "property_type", "postal_code", "loan_number", "loan_purpose", "loan_term",
            "number_of_borrowers", "seller_name", "servicer_name", "super_conforming",
            "prerelief_loan_number", "program_indicator", "relief_refinance",
            "valuation_method", "interest_only", "mi_cancellation"]
orig_dtypes = {
    "credit_score": 'Float32', 
    "date_first_payment": 'string', # YYYYMM
    "first_time_homebuyer": 'string',
    "date_maturity": 'string', # YYYYMM
    "MSA": 'string', # CLEANED
    "insurance_percent": 'Float32', 
    "nr_units": 'Int32', 
    "occupancy": 'string', 
    "original_combined_loan_to_value": 'Float32', # what is HARP?
    "original_debt_to_income": 'Float32', 
    "original_upb": 'Float32', # UPB = Unpaid principle balance
    "original_loan_to_value": 'Float32', 
    "original_interest_rate": 'Float32', 
    "channel": 'string', 
    "prepayment_penalty": 'string', 
    "amortization_type": 'string', 
    "property_state": 'string', # Remove Alaska / Hawaii ?
    "property_type": 'string', 
    "postal_code": 'string', 
    "loan_number": 'string', 
    "loan_purpose": 'string', 
    "loan_term": 'Int32', # Number of months
    "number_of_borrowers": 'Int32', 
    "seller_name": 'string', 
    "servicer_name": 'string', 
    "super_conforming": 'string', # CLEANED
    "prerelief_loan_number": 'string', 
    "program_indicator": 'string', 
    "relief_refinance": 'string', # CLEANED
    "valuation_method": 'Int32', 
    "interest_only": 'string', 
    "mi_cancellation": 'string', # Here 7 is NA, 9 is not disclosed (is it important these are treated different?)
}
orig_na = {
    "credit_score": 9999,
    "first_time_homebuyer": '9',
    "insurance_percent": 999,
    "nr_units": 99,
    "occupancy": '9',
    "original_combined_loan_to_value": 999,
    "original_debt_to_income": 999, 
    "original_loan_to_value": 999, 
    "channel": '9',
    "property_type": '99',
    "postal_code": '00', 
    "loan_purpose": '9',
    "number_of_borrowers": 99, 
    "program_indicator": '9', 
    "valuation_method": 9, 
    "mi_cancellation": '9'
}

# Performance dataset specs
svcg_cols = ["loan_number", "period", "current_upb", "current_loan_delinquency_status",
                "loan_age", "remaining_months", "date_defect_settlement", "modified",
                "zero_balance_code", "date_zero_balance", "current_interest_rate",
                "current_deferred_UPB", "DDLPI", "mi_recoveries", "net_sale_proceeds",
                "non_mi_recoveries", "expenses", "legal_costs", "maintenance_costs",
                "taxes_and_insurance", "miscellaneous_expenses", "actual_loss",
                "modification_cost", "step_modification", "deferred_payment_plan",
                "estimated_loan_to_value", "zero_balance_removal_UPB",
                "delinquent_interest", "delinquent_desaster", "assistance_code",
                "month_modification_cost", "interest_bearing_UPB"]
svcg_dtypes = {
    "loan_number": 'string', 
    "period": 'string', # YYYYMM 
    "current_upb": 'Float32',
    "current_loan_delinquency_status": 'string',
    "loan_age": "Int32", # Number of scheduled payments
    "remaining_months": "Int32", # Number of months
    "date_defect_settlement": "string", # YYYYMM
    "modified": 'string',
    "zero_balance_code": 'string', 
    "date_zero_balance": 'string', # YYYMM
    "current_interest_rate": 'Float32',
    "current_deferred_UPB": 'Float32',
    "DDLPI": 'Int32',
    "mi_recoveries": 'Float32',
    "net_sale_proceeds": 'Float32', # This should be a float
    "non_mi_recoveries": 'Float32',
    "expenses": 'Float32',
    "legal_costs": 'Float32',
    "maintenance_costs": 'Float32',
    "taxes_and_insurance": 'Float32',
    "miscellaneous_expenses": 'Float32',
    "actual_loss": 'Float32',
    "modification_cost": 'Float32',
    "step_modification": 'string',
    "deferred_payment_plan": 'string',
    "estimated_loan_to_value": 'Int32',
    "zero_balance_removal_UPB": 'Float32',
    "delinquent_interest": 'Float32',
    "delinquent_desaster": 'string', # Typo in column name
    "assistance_code": 'string',
    "month_modification_cost": 'Float32',
    "interest_bearing_UPB": 'Float32'
}
svcg_na = {
    "modified": 'Null',
    "net_sale_proceeds": 'U',
    "step_modification": 'Null',
    "deferred_payment_plan": 'Null',
    "estimated_loan_to_value": '999',
    "delinquent_desaster": 'Null',
    "assistance_code": 'Null',
}

DATA_DIR = "PATH TO FULL SFLLD"


def load_data(q):
    print(f"Importing data for {q}")

    # DATA_DIR = "data/mortgage_data/full_data/"

    # Mortgage origination data
    quarterly_orig_df = pd.read_csv(
        DATA_DIR + f"historical_data_{q}.txt", 
        sep='|', 
        header=None,
        names=orig_cols, 
        dtype=orig_dtypes, 
        na_values=orig_na
    ) 
    # Filter for only single-family (SF) fixed-rate-mortgage (FRM) 30-year contracts
    quarterly_orig_df = quarterly_orig_df[
        (quarterly_orig_df["property_type"] == "SF") &
        (quarterly_orig_df["amortization_type"] == "FRM") &
        (quarterly_orig_df["loan_term"] == 360)
    ]

    # Adding to main origination dataframe
    print("Number of new origination entries:", len(quarterly_orig_df))

    # Mortgage performance data
    quarterly_svcg_df = pd.read_csv(
        DATA_DIR + f"historical_data_time_{q}.txt", 
        sep='|',
        header=None,
        names=svcg_cols, 
        dtype=svcg_dtypes, 
        na_values=svcg_na
    )

    quarterly_svcg_df = quarterly_svcg_df[
        quarterly_svcg_df["loan_number"].isin(quarterly_orig_df["loan_number"])
    ]
    
    print("Number of new performance entries:", len(quarterly_svcg_df))

    return quarterly_orig_df, quarterly_svcg_df

def get_default_data(orig_df, svcg_df, default_def):
    # Filtering out inactive loans for the first 3 months
    activity_mask = (svcg_df['loan_age'].isin([0, 1, 2, 3])) & (svcg_df['modified'].isnull())

    # Select the loan_numbers that meet the criteria
    valid_loans = svcg_df.loc[activity_mask, 'loan_number']

    # Filter orig_data using those loan_numbers
    orig_df = orig_df[orig_df['loan_number'].isin(valid_loans)]
    svcg_df = svcg_df[svcg_df['loan_number'].isin(orig_df['loan_number'])]

    # Identifying defaulted loans by loan number, earliest period
    # Filter for rows where delinquency status is >= 3
    # nondefault_codes = ['0', '1', '2', 'RA']
    nondefault_codes = ['0', '1', '2']
    if default_def == '180D':
        nondefault_codes = ['0', '1', '2', '3', '4', '5']
    
    defaulted_svcg_rows = svcg_df[~svcg_df['current_loan_delinquency_status'].isin(nondefault_codes)]

    defaulted_ids = defaulted_svcg_rows['loan_number'].unique()
    defaulted_orig_df = orig_df[orig_df['loan_number'].isin(defaulted_ids)]
    defaulted_svcg_df = svcg_df[svcg_df['loan_number'].isin(defaulted_ids)]

    # Group by loan_number and get the earliest period (min)
    first_default_idx = defaulted_svcg_rows.groupby('loan_number')['period'].idxmin()
    # Dataframe for data pertaining to loss incurred on (defaulted) mortgage
    defaulted_lgd_df = defaulted_svcg_rows.loc[first_default_idx, ['loan_number', 'period', 'current_upb']] # Extract period and UPB at time of earliest default
    defaulted_lgd_df.columns = ["loan_number", "date_default", "upb_at_default"]
    zerobalancedates = defaulted_svcg_df.loc[defaulted_svcg_df["date_zero_balance"].notna(),
                        ["loan_number", "date_zero_balance"]]

    # Left merge onto orig_data
    defaulted_lgd_df = defaulted_lgd_df.merge(zerobalancedates, on="loan_number", how="left")

    return defaulted_ids, defaulted_orig_df, defaulted_svcg_df, defaulted_lgd_df

def clean_process_data(orig_df, svcg_df, lgd_df):
    postcode_data = pd.read_csv("shapefiles/ZIP_centroids.csv")

    # Restrict to mainland
    postcode_data = postcode_data[
        (postcode_data["X_centroid"] < -66) &
        (postcode_data["X_centroid"] > -125) &
        (postcode_data["Y_centroid"] > 24) &
        (postcode_data["Y_centroid"] < 50)
    ]

    # Remove ZIP3 == 1
    postcode_data = postcode_data[postcode_data["ZIP3"] != 1]

    # Removing loans with NA postal code
    orig_df = orig_df.loc[orig_df['postal_code'].notna(), :]

    # Extract ZIP3 from orig_data$postal_code (remove last 2 characters)
    orig_df.loc[:, "ZIP3"] = orig_df["postal_code"].str[:-2].astype(int)

    # Merge with centroid data
    orig_df = orig_df.merge(
        postcode_data[["ZIP3", "X_centroid", "Y_centroid"]],
        on="ZIP3",
        how="inner" 
    )

    # Removing performance / lgd entries with invalid ZIP3s
    svcg_df = svcg_df.loc[svcg_df['loan_number'].isin(orig_df['loan_number']), :]
    lgd_df = lgd_df.loc[lgd_df['loan_number'].isin(orig_df['loan_number']), :]

    orig_df["ZIP3"] = orig_df["ZIP3"].astype(str).str.zfill(3)

    # Data cleaning
    orig_df.loc[orig_df['relief_refinance'].isna(), 'relief_refinance'] = "0"
    orig_df.loc[orig_df['relief_refinance'] == "Y", 'relief_refinance'] = "1"

    # --- first_time_homebuyer ---
    orig_df.loc[orig_df['first_time_homebuyer'] == "N", 'first_time_homebuyer'] = "0"
    orig_df.loc[orig_df['first_time_homebuyer'] == "Y", 'first_time_homebuyer'] = "1"

    # --- MSA ---
    orig_df['MSA'] = orig_df['MSA'].notna().astype("Int32")

    # --- super_conforming ---
    orig_df.loc[orig_df['super_conforming'].isna(), 'super_conforming'] = "0"
    orig_df.loc[orig_df['super_conforming'] == "Y", 'super_conforming'] = "1"
    orig_df['super_conforming'] = orig_df['super_conforming'].astype("Int32")

    # --- number_of_borrowers ---
    orig_df.loc[orig_df['number_of_borrowers'].isin([3, 4]), 'number_of_borrowers'] = 2

    # Impute the data
    # Mode helper function
    def Mode(s):
        return s.mode(dropna=True).iloc[0]

    # --- credit_score ---
    impute_val = orig_df["credit_score"].mean(skipna=True)
    orig_df["credit_score"] = orig_df["credit_score"].fillna(value=impute_val)

    # --- first_time_homebuyer ---
    impute_val = Mode(orig_df["first_time_homebuyer"])
    orig_df["first_time_homebuyer"] = orig_df["first_time_homebuyer"].fillna(value=impute_val)

    # --- insurance_percent ---
    impute_val = orig_df["insurance_percent"].mean(skipna=True)
    orig_df["insurance_percent"] = orig_df["insurance_percent"].fillna(value=impute_val)

    # --- nr_units ---
    impute_val = Mode(orig_df["nr_units"])
    orig_df["nr_units"] = orig_df["nr_units"].fillna(value=impute_val)

    # --- original_combined_loan_to_value ---
    impute_val = orig_df["original_combined_loan_to_value"].mean(skipna=True)
    orig_df["original_combined_loan_to_value"] = orig_df["original_combined_loan_to_value"].fillna(value=impute_val)

    # --- original_debt_to_income ---
    impute_val = orig_df["original_debt_to_income"].mean(skipna=True)
    orig_df["original_debt_to_income"] = orig_df["original_debt_to_income"].fillna(value=impute_val)

    # --- original_loan_to_value ---
    impute_val = orig_df["original_loan_to_value"].mean(skipna=True)
    orig_df["original_loan_to_value"] = orig_df["original_loan_to_value"].fillna(value=impute_val)

    # --- channel ---
    impute_val = Mode(orig_df["channel"])
    orig_df["channel"] = orig_df["channel"].fillna(value=impute_val)

    # --- number_of_borrowers ---
    impute_val = Mode(orig_df["number_of_borrowers"])
    orig_df["number_of_borrowers"] = orig_df["number_of_borrowers"].fillna(value=impute_val)


    # Convert categorical variables to factors
    factor_cols = ["occupancy", "loan_purpose", "first_time_homebuyer", 
            "MSA", "channel", "number_of_borrowers", "relief_refinance"] # "nr_units",
    orig_df[factor_cols] = orig_df[factor_cols].astype("category")


    # Converting to DateTime 
    # --- Convert YYYYMM → YYYYMM01 and parse as dates ---
    orig_df.loc[:, "date_first_payment"] = pd.to_datetime(orig_df["date_first_payment"] + "01", format="%Y%m%d")
    orig_df.loc[:, "date_maturity"]      = pd.to_datetime(orig_df["date_maturity"] + "01", format="%Y%m%d")
    lgd_df.loc[:, "date_default"]       = pd.to_datetime(lgd_df["date_default"] + "01", format="%Y%m%d", errors="coerce")
    lgd_df.loc[:, "date_zero_balance"]  = pd.to_datetime(lgd_df["date_zero_balance"] + "01", format="%Y%m%d", errors="coerce")

    # --- Loan start date = first_payment minus 1 month ---
    orig_df["date_loan_start"] = orig_df["date_first_payment"].apply(
        lambda d: d - relativedelta(months=1)
    )

    # --- Loan end date = min(default, zero_balance, maturity) ---
    defaulted_lifespan_df = orig_df.merge(lgd_df[["loan_number", "date_default", "date_zero_balance"]], on="loan_number", how="left")
    orig_df["date_loan_end"] = defaulted_lifespan_df[
        ["date_default", "date_zero_balance", "date_maturity"]
    ].min(axis=1)   # Pandas respects NaN, identical to R's min(..., na.rm=TRUE)

    # Converting period for performance dataset entries
    svcg_df.loc[:, "period"] = pd.to_datetime(svcg_df["period"] + "01", format="%Y%m%d")

    return orig_df, svcg_df, lgd_df

def get_ZB2_data(D_ids, orig_df, svcg_df, lgd_df):
    # D_ids = defaulted_ids #default_data['loan_number']
    print(f"Number of unique defaulted loans: |D| = {len(D_ids)}")

    ZB_entry = svcg_df[svcg_df['zero_balance_code'].notna()]
    # ZB_ids = ZB_entry['loan_number'].unique()
    # print(f"Number of unique defaulted loans with ZBC: |ZB| = {len(ZB_ids)}")

    ZB1_codes = ['01']
    ZB1_entry = ZB_entry[ZB_entry['zero_balance_code'].isin(ZB1_codes)] # SVCG entry with zero balance code (final entry)
    # ZB1_ids = ZB1_entry['loan_number'].unique()
    # print(f"Number of unique defaulted loans in ZBC1: |ZBC_1| = {len(ZB1_ids)}")

    ZB2_codes = ['02', '03', '09', '15'] # CODES FOR WHICH actual_loss IS AVAILABLE
    ZB2_entry = ZB_entry[ZB_entry['zero_balance_code'].isin(ZB2_codes)]
    # ZB2_ids = ZB2_entry['loan_number'].unique()
    # print(f"Number of unique defaulted loans in ZBC2: |ZBC_2| = {len(ZB2_ids)}")

    ZB3_codes = ['16', '96']
    ZB3_entry = ZB_entry[ZB_entry['zero_balance_code'].isin(ZB3_codes)]
    # ZB3_ids = ZB3_entry['loan_number'].unique()
    # print(f"Number of unique defaulted loans in ZBC3: |ZBC_3| = {len(ZB3_ids)}")

    ZB2usable_entry = ZB2_entry[ZB2_entry['actual_loss'].notna()]
    ZB2usable_ids = ZB2usable_entry['loan_number'].unique()
    print(f"Number of unique defaulted loans in ZBC2 with actual_loss defined: |ZBC_2'| = {len(ZB2usable_ids)}")

    ZB2_orig = orig_df[orig_df['loan_number'].isin(ZB2usable_ids)]
    ZB2_svcg = svcg_df[svcg_df['loan_number'].isin(ZB2usable_ids)]
    ZB2_lgd = lgd_df[lgd_df['loan_number'].isin(ZB2usable_ids)]

    lgd_data = ZB2usable_entry[['loan_number', 'actual_loss', 'net_sale_proceeds', 'zero_balance_removal_UPB', 'non_mi_recoveries', 'expenses', 'delinquent_interest']]
    ZB2_lgd = ZB2_lgd.merge(lgd_data, on="loan_number", how='left')

    return ZB2usable_ids, ZB2_orig, ZB2_svcg, ZB2_lgd

def calculate_LGD(lgd_df):
    # LGD0: Uses only net_sales
    def LGD0(net_sale_proceeds, upb_at_default, zero_balance_removal_upb):
        try:
            return (zero_balance_removal_upb - net_sale_proceeds) / upb_at_default
        except:
            return 1 - net_sale_proceeds / zero_balance_removal_upb

    # LGD1: actual_loss without accrued interest (carrying costs)
    def LGD1(actual_loss, delinquent_interest, upb_at_default, zero_balance_removal_upb):
        try:
            return (actual_loss - delinquent_interest) / upb_at_default 
        except:
            return (actual_loss - delinquent_interest) / zero_balance_removal_upb

    # LGD2: Using actual_loss directly (over UPB at default)
    def LGD2(actual_loss, upb_at_default, zero_balance_removal_upb):
        try:
            return actual_loss / upb_at_default 
        except:
            return actual_loss / zero_balance_removal_upb

    # Loss-severity taken from Philly Fed paper (no carrying costs)
    def LS(net_sale_proceeds, zero_balance_removal_upb, upb_at_default, expenses, nonMI_recoveries):
        try:
            return (zero_balance_removal_upb + expenses - net_sale_proceeds - nonMI_recoveries) / zero_balance_removal_upb
        except:
            return (zero_balance_removal_upb + expenses - net_sale_proceeds - nonMI_recoveries) / upb_at_default


    LGD0_helper = lambda df: LGD0(df['net_sale_proceeds'], df['upb_at_default'], df['zero_balance_removal_UPB'])
    lgd_df['LGD0'] = lgd_df.apply(LGD0_helper, axis=1)

    LGD1_helper = lambda df: LGD1(-df['actual_loss'], df['delinquent_interest'], df['upb_at_default'], df['zero_balance_removal_UPB'])
    lgd_df['LGD1'] = lgd_df.apply(LGD1_helper, axis=1)

    # NOTE: 'actual_loss' NEGATED BELOW TO MAKE IT CONSISTENT WITH FORMULA IN USER GUIDE
    LGD2_helper = lambda df: LGD2(-df['actual_loss'], df['upb_at_default'], df['zero_balance_removal_UPB'])
    lgd_df['LGD2'] = lgd_df.apply(LGD2_helper, axis=1)

    LS_helper = lambda df: LS(df['net_sale_proceeds'], df['zero_balance_removal_UPB'], df['upb_at_default'], -df['expenses'], df['non_mi_recoveries'])
    lgd_df['loss_severity'] = lgd_df.apply(LS_helper, axis=1)

    return lgd_df


def get_quarterly_defaults(quarter_name):
    orig_df, svcg_df = load_data(quarter_name)

    default_definition = ['90D', '180D'] # CHANGE DEPENDING ON DEFINITION USED

    default_ids, default_orig_df, default_svcg_df, default_lgd_df = get_default_data(orig_df, svcg_df, default_definition[1])

    default_orig_df, default_svcg_df, default_lgd_df = clean_process_data(default_orig_df, default_svcg_df, default_lgd_df)

    ZB2_ids, ZB2_orig, ZB2_svcg, ZB2_lgd = get_ZB2_data(default_ids, default_orig_df, default_svcg_df, default_lgd_df)
    N_ids = len(ZB2_ids)

    ZB2_lgd = calculate_LGD(ZB2_lgd)

    return N_ids, ZB2_orig, ZB2_svcg, ZB2_lgd

def extract_yearly_defaults(year_name):
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    annual_orig_df, annual_svcg_df, annual_lgd_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    N_annual = 0

    for q in quarters:
        N_quarterly, q_orig_df, q_svcg_df, q_lgd_df = get_quarterly_defaults(year_name + q)

        annual_orig_df = pd.concat([annual_orig_df, q_orig_df], ignore_index=True)
        annual_svcg_df = pd.concat([annual_svcg_df, q_svcg_df], ignore_index=True)
        annual_lgd_df = pd.concat([annual_lgd_df, q_lgd_df], ignore_index=True)

        N_annual += N_quarterly
    
    return N_annual, annual_orig_df, annual_svcg_df, annual_lgd_df
