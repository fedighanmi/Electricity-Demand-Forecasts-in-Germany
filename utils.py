import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import collections
import math
from scipy.stats import t
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# The below is a function to help in datetime subsetting
def date_finder(date, time):
    date_str = date + " " + time
    date_format = "%Y-%m-%d %H:%M:%S"

    return datetime.strptime(date_str, date_format)


def clean_data_1(df, holiday_included=True, co2_included=True, temp_included=True, gas_included=True,
               trade_load_included=True, meteologica_included=True, generation_per_output_included=True):
    # Here we do basic changes like change column names and change date format to datetime

    df['Datum von'] = pd.to_datetime(df['Datum von'], dayfirst=True)
    df.rename(columns={'Gesamt (Netzlast) [MWh] Berechnete Auflösungen': 'total_load',
                       'Residuallast [MWh] Berechnete Auflösungen': 'residual_load',
                       'Pumpspeicher [MWh] Berechnete Auflösungen': 'hydro_storage'}, inplace=True)

    # Here we replace characters so that when we make numerical columns float, no error arises.

    df['total_load'] = df['total_load'].str.replace('.', '', regex=True)
    df['total_load'] = df['total_load'].str.replace(',', '.', regex=True)

    df['total_load'] = df['total_load'].astype(float)

    date_str = "2024-3-15 23:00:00"
    date_format = "%Y-%m-%d %H:%M:%S"

    last_date = datetime.strptime(date_str, date_format)

    # We delete data after 15th March 2024.
    df_subset = df.loc[df["Datum von"] <= last_date, :]

    # We sort our dates
    df_sorted = df_subset.sort_values(by='Datum von')

    # Create date range without DST
    date_range = pd.date_range(start='2015-01-01 00:00:00', end='2024-03-15 23:00:00', freq='H')
    df_utc = pd.DataFrame({'datetime_clean': date_range})

    # Merge both dataframes
    df_merged = pd.merge(df_sorted, df_utc, left_index=True, right_index=True)

    df_merged['total_load_clean'] = pd.Series(dtype='float64')
    df_merged['residual_load_clean'] = pd.Series(dtype='float64')
    df_merged['hydro_storage_clean'] = pd.Series(dtype='float64')

    # Here the code might take some time to run as it is transforming our data to make it without the daylight
    # savings time. I made it without the use of external packages to have control over what is happening

    summer_time = False
    for index, row in df_merged.iterrows():
        date_dirty = row['Datum von']
        date_clean = row['datetime_clean']

        if (date_dirty == date_clean) and (summer_time == False):
            df_merged.at[index, 'total_load_clean'] = row['total_load']
            df_merged.at[index, 'residual_load_clean'] = row['residual_load']
            df_merged.at[index, 'hydro_storage_clean'] = row['hydro_storage']

        elif (date_dirty == date_clean) and (summer_time == True):
            df_merged.at[index, 'total_load_clean'] = df_merged.at[index - 1, 'total_load']
            df_merged.at[index, 'residual_load_clean'] = df_merged.at[index - 1, 'residual_load']
            df_merged.at[index, 'hydro_storage_clean'] = df_merged.at[index - 1, 'hydro_storage']
            summer_time = False

        else:
            df_merged.at[index, 'total_load_clean'] = df_merged.at[index - 1, 'total_load']
            df_merged.at[index, 'residual_load_clean'] = df_merged.at[index - 1, 'residual_load']
            df_merged.at[index, 'hydro_storage_clean'] = df_merged.at[index - 1, 'hydro_storage']
            summer_time = True

    # extract = df_merged[["datetime_clean", "total_load_clean", "residual_load_clean","hydro_storage_clean"]]

    oct_indices_to_delete = [7130, 16034, 24770, 33506, 42242, 50978, 59882, 68617, 77354]

    extract = df_merged[["datetime_clean", "total_load_clean"]]

    # for each column, we added an if statement to control the code that will run when the
    # function is called.

    # This code below will create the holiday column when it gets executed.

    if holiday_included:
        german_holidays = holidays.country_holidays('DE')
        extract["holiday"] = extract["datetime_clean"].apply(lambda x: 1 if x in german_holidays else 0)

    # This code below will create the co2 column when it gets executed.

    if co2_included:
        eua = pd.read_csv("external_data/co2_eex_eua_futures.csv")
        columns_to_drop_co2 = ["Unnamed: 0", "DELIVERY_DATE"]
        co2 = eua.drop(columns=columns_to_drop_co2)
        co2['TRADING_DATE'] = pd.to_datetime(co2['TRADING_DATE'])
        co2 = co2.sort_values(by='TRADING_DATE')

        co2_idx = co2.set_index('TRADING_DATE')
        co2_new = co2_idx.resample('D').mean()
        co2_new.reset_index(inplace=True)
        co2_new = co2_new.loc[co2_new["TRADING_DATE"] <= date_finder("2024-03-15", "00:00:00"), :]
        co2_new = pd.DataFrame(np.repeat(co2_new.values, 24, axis=0), columns=co2_new.columns)

        co2_new['MEAN'] = co2_new['MEAN'].astype(float)
        extract["co2_prices"] = co2_new["MEAN"]

    # This code below will create the temperature column when it gets executed.

    if temp_included:
        temp = pd.read_csv("external_data/gfs_weather.csv")
        temp = temp.loc[temp["Country"] == "DE", :]
        columns_to_drop_temp = ["Unnamed: 0", "DeliveryDateUtc", "Country"]
        filtered_temp = temp.drop(columns=columns_to_drop_temp)
        filtered_temp['TradingDateUtc'] = pd.to_datetime(filtered_temp['TradingDateUtc'])
        temp_clean = filtered_temp.sort_values(by='TradingDateUtc')

        temp_clean = temp_clean.loc[temp_clean["TradingDateUtc"] <= date_finder("2024-03-15", "23:00:00"), :]
        temp_clean = temp_clean.loc[temp_clean["TradingDateUtc"] >= date_finder("2015-01-01", "00:00:00"), :]

        temp_idx = temp_clean.set_index('TradingDateUtc')
        temp_agg = temp_idx.resample('D').mean()
        temp_agg.reset_index(inplace=True)

        temp_agg = pd.DataFrame(np.repeat(temp_agg.values, 24, axis=0), columns=temp_agg.columns)

        temp_agg['OperationalRun'] = temp_agg['OperationalRun'].astype(float)

        extract["temp"] = temp_agg["OperationalRun"]

    # This code below will create the gas prices column when it gets executed.

    if gas_included:
        gas = pd.read_csv("external_data/gas_eex_eua_futures.csv")
        # gas = gas.loc[gas["Sub-Region"] == "GPL (formerly BEB/GUD)",: ]
        columns_to_drop_gas = ['DeliveryDateUtc', 'delivery_period_name', 'Commodity', 'Region',
                               'Sub-Region', "Unnamed: 0"]

        # Drop the columns
        gas = gas.drop(columns=columns_to_drop_gas)
        gas['TradingDateUtc'] = pd.to_datetime(gas['TradingDateUtc'])
        gas = gas.sort_values(by='TradingDateUtc')

        gas = gas.loc[gas["TradingDateUtc"] >= date_finder("2015-01-01", "00:00:00"), :]
        gas = gas.loc[gas["TradingDateUtc"] <= date_finder("2024-03-15", "00:00:00"), :]

        gas_idx = gas.set_index('TradingDateUtc')
        gas_new = gas_idx.resample('D').mean()
        gas_new.reset_index(inplace=True)

        new_row = pd.DataFrame({'TradingDateUtc': ['2015-01-01'], 'SettlementClearing': [20.728833]})
        new_row['TradingDateUtc'] = pd.to_datetime(new_row['TradingDateUtc'])
        gas_new = pd.concat([new_row, gas_new]).reset_index(drop=True)

        gas_new = gas_new.fillna(method='ffill')
        gas_new = pd.DataFrame(np.repeat(gas_new.values, 24, axis=0), columns=gas_new.columns)

        gas_new['SettlementClearing'] = gas_new['SettlementClearing'].astype(float)

        extract["gas_prices"] = gas_new["SettlementClearing"]

    # This code below will create the net_trade column when it gets executed.

    if trade_load_included:
        trade_load = pd.read_csv("external_data/physical_power_flow_2015_2024_hourly.csv", delimiter=";")
        trade_load.replace("-", "0", inplace=True)

        columns_to_exclude_trade_load = ["Datum von", "Datum bis", "Nettoexport [MWh] Berechnete Auflösungen"]
        trade_load_c = trade_load.drop(columns=columns_to_exclude_trade_load)
        trade_load = trade_load[["Datum von"]]

        trade_load_c = trade_load_c.applymap(lambda x: convert_to_float(x))
        trade_load_c["net_trade"] = trade_load_c.sum(axis=1)
        trade_load["net_trade"] = trade_load_c["net_trade"]

        trade_load['Datum von'] = pd.to_datetime(trade_load['Datum von'])
        trade_load = trade_load.sort_values(by='Datum von')

        trade_load = trade_load.loc[trade_load["Datum von"] <= date_finder("2024-03-15", "23:00:00"), :]
        trade_load = trade_load.loc[trade_load["Datum von"] >= date_finder("2015-01-01", "00:00:00"), :]

        # Here we delete the DST instances for the clock backward.

        trade_load = trade_load.drop(oct_indices_to_delete)

        # in the code below, the first instance of 1am will automatically be duplicated for 2am with the
        # fill forward imputation as the resample function will result in hourly data without gaps.

        trade_load_idx = trade_load.set_index('Datum von')
        trade_load_agg = trade_load_idx.resample('H').mean()
        trade_load_agg.reset_index(inplace=True)

        trade_load_agg = trade_load_agg.fillna(method='ffill')

        extract["net_trade"] = trade_load_agg["net_trade"]

    # This code below will create the wind and solar column forecasts when it gets executed.

    if meteologica_included:
        # The amount of code may be significant due to the need to preprocess.

        meteo = pd.read_csv("external_data/meteologica.csv")

        # We filter based on german providers

        area_to_keep = ["DE", "tennet", "amprion", "50hertz", "transnetbw"]
        filtered_meteo = meteo[meteo['Area'].isin(area_to_keep)]

        # We aggregate to to each hour having same provider and Technology (Solar or wind) and choose our
        # Time intervals

        sum_by_area = filtered_meteo.groupby(['TradingDateUtc', 'Area', 'Technology']).agg(
            {'Mean': 'mean'}).reset_index()
        so = sum_by_area.loc[sum_by_area["Technology"] == "solar", :]
        so['TradingDateUtc'] = pd.to_datetime(so['TradingDateUtc'])
        so = so.sort_values(by='TradingDateUtc')
        so = so.loc[so["TradingDateUtc"] <= date_finder("2024-03-15", "00:00:00"), :]
        so = so.loc[so["TradingDateUtc"] >= date_finder("2015-01-01", "00:00:00"), :]

        # We make our data fully hourly in that time range (It will also delete the DST automatically)
        # for the clock forward and using forward fill, it will take 1am value.

        so_idx = so.set_index('TradingDateUtc')
        so_new = so_idx.resample('H').mean(numeric_only=True)
        so_new.reset_index(inplace=True)

        # Some hours of days are missing, so we add them manually

        new_row_beg = pd.DataFrame({'TradingDateUtc': ['2015-01-01 00:00:00'], 'Mean': [417.858315]})
        new_row_end = pd.DataFrame({'TradingDateUtc': ['2024-03-15 23:00:00'], 'Mean': [5652.233645]})
        new_row_beg['TradingDateUtc'] = pd.to_datetime(new_row_beg['TradingDateUtc'])
        new_row_end['TradingDateUtc'] = pd.to_datetime(new_row_end['TradingDateUtc'])
        so_new_2 = pd.concat([so_new, new_row_beg]).reset_index(drop=True)
        so_new_3 = pd.concat([so_new_2, new_row_end]).reset_index(drop=True)
        so_new_3['TradingDateUtc'] = pd.to_datetime(so_new_3['TradingDateUtc'])
        so_new_3 = so_new_3.sort_values(by='TradingDateUtc')

        # We sort again

        so_idx = so_new_3.set_index('TradingDateUtc')
        so_new = so_idx.resample('H').mean(numeric_only=True)
        so_new.reset_index(inplace=True)

        # We use Last obersavtion carried forward interpolation

        so_new = so_new.fillna(method='ffill')

        # We sum the consumption of each provider together with others and sort our data again based
        # on datetime.

        wi = sum_by_area.loc[sum_by_area["Technology"] == "wind", :]
        wi_summed = wi.groupby(['TradingDateUtc']).agg({'Mean': 'sum'}).reset_index()
        wi_summed['TradingDateUtc'] = pd.to_datetime(wi_summed['TradingDateUtc'])
        wi_summed = wi_summed.sort_values(by='TradingDateUtc')

        wi_summed = wi_summed.loc[wi_summed["TradingDateUtc"] <= date_finder("2024-03-15", "00:00:00"), :]

        wi_summed_idx = wi_summed.set_index('TradingDateUtc')
        wi_new = wi_summed_idx.resample('D').mean(numeric_only=True)
        wi_new.reset_index(inplace=True)

        wi_new = wi_new.fillna(method='ffill')

        wi_new['TradingDateUtc'] = pd.to_datetime(wi_new['TradingDateUtc'])
        wi_new = wi_new.sort_values(by='TradingDateUtc')

        wi_new_idx = wi_new.set_index('TradingDateUtc')
        wi_final = wi_new_idx.resample('H').mean()
        wi_final.reset_index(inplace=True)

        wi_final = wi_final.fillna(method='ffill')

        # Add the missing data of the wind and repeat the whole preprocessing process that we
        # did above

        wind_supp = pd.read_csv("external_data/forecasted_generation_day_ahead_2015_2016_hourly.csv", delimiter=";")

        wind_supp = wind_supp.loc[:, ["Start date", "Wind offshore [MWh] Calculated resolutions",
                                      "Wind onshore [MWh] Calculated resolutions"]]
        wind_supp['Start date'] = pd.to_datetime(wind_supp['Start date'])
        wind_supp = wind_supp.sort_values(by='Start date')
        wind_supp.replace("-", np.nan, inplace=True)
        wind_supp = wind_supp.fillna(method='ffill')

        wind_supp['Wind offshore [MWh] Calculated resolutions'] = wind_supp[
            'Wind offshore [MWh] Calculated resolutions'].str.replace(',', '', regex=True)

        wind_supp['Wind onshore [MWh] Calculated resolutions'] = wind_supp[
            'Wind onshore [MWh] Calculated resolutions'].str.replace(',', '', regex=True)

        wind_supp['Wind offshore [MWh] Calculated resolutions'] = wind_supp[
            'Wind offshore [MWh] Calculated resolutions'].astype(float)
        wind_supp['Wind onshore [MWh] Calculated resolutions'] = wind_supp[
            'Wind onshore [MWh] Calculated resolutions'].astype(float)

        wind_supp["Mean"] = wind_supp['Wind onshore [MWh] Calculated resolutions'] + wind_supp[
            'Wind offshore [MWh] Calculated resolutions']

        wind_supp = wind_supp.loc[:, ["Start date", "Mean"]]
        wind_supp.rename(columns={'Start date': 'TradingDateUtc'}, inplace=True)

        stacked_wind = pd.concat([wind_supp, wi_final], axis=0)

        new_row = pd.DataFrame({'TradingDateUtc': ['2024-03-15 23:00:00'], 'Mean': [18114.467290]})
        new_row['TradingDateUtc'] = pd.to_datetime(new_row['TradingDateUtc'])
        stacked_wind = pd.concat([stacked_wind, new_row]).reset_index(drop=True)

        # Stack both dataframes

        stacked_wind_idx = stacked_wind.set_index('TradingDateUtc')
        stacked_wind_new = stacked_wind_idx.resample('H').mean()
        stacked_wind_new.reset_index(inplace=True)

        stacked_wind_new = stacked_wind_new.fillna(method='ffill')

        # Add preprocessed columns to the extract data to work with later

        extract["wind_forecast"] = stacked_wind_new["Mean"]
        extract["solar_forecast"] = so_new["Mean"]

    # This code below will create the generation per unit column for each unit when it gets executed.

    if generation_per_output_included:
        # The data was previously preprocessed using R code attached with this Jupyter notebook.

        Gen = pd.read_csv("external_data/generation_output_per_type_clean.csv")
        Gen = Gen.drop(["Unnamed: 0", "ConsumptionSum_Biomass",
                        "ConsumptionSum_Fossil",
                        "ConsumptionSum_Geothermal",
                        "ConsumptionSum_Hydro"
                           , "ConsumptionSum_Other"], axis=1)
        Gen['DateTime'] = pd.to_datetime(Gen['DateTime'])
        Gen = Gen.sort_values(by='DateTime')
        Gen = Gen.iloc[:80688]

        Gen.reset_index(inplace=True)
        Gen.drop(["index", "DateTime"], inplace=True, axis=1)

        Gen.rename(columns={'GenerationOutputSum_Biomass': 'biomass_output',
                            'GenerationOutputSum_Fossil': 'fossil_output',
                            'GenerationOutputSum_Geothermal': 'geothermal_output',
                            "GenerationOutputSum_Hydro": "hydro_output",
                            "GenerationOutputSum_Other": "other_output"
                            }, inplace=True)

        extract_final = pd.concat([extract, Gen], axis=1)

    # extract_final.to_csv('all_data.csv', index=False)

    return extract_final.sort_values(by='datetime_clean')


# The below function is to subset our forecasts to the dates between 8pm and 8apm.
def res_subset(ts_data, actual_values, forecast, latest_lag=3):
    df_res_subset = pd.DataFrame(columns=['actual', 'forecast'])

    actual_values_subset = pd.DataFrame(columns=['date', 'value'])
    forecast_subset = pd.DataFrame(columns=['date', 'value'])

    date_format = "%H:%M:%S"

    date_start = "20:00:00"
    date_end = "08:00:00"

    for idx in range(0, len(ts_data[latest_lag:])):

        if idx == 20000:
            print("25% Complete")
        elif idx == 40000:
            print("50% Complete")
        elif idx == 60000:
            print("75% Complete")
        elif idx == 80660:
            print("100% Complete")
        else:
            pass

        if (actual_values.index[idx].time() >= datetime.strptime(date_start, date_format).time()) or (
                actual_values.index[idx].time() <= datetime.strptime(date_end, date_format).time()):
            actual_values_subset.at[idx, "date"] = actual_values.index[idx]
            actual_values_subset.at[idx, "value"] = actual_values[idx]

            forecast_subset.at[idx, "date"] = forecast.index[idx]
            forecast_subset.at[idx, "value"] = forecast[idx]

    df_res_subset["actual"] = actual_values_subset["value"]
    df_res_subset["forecast"] = forecast_subset["value"]
    df_res_subset.index = actual_values_subset["date"]

    return df_res_subset


# The below function will convert to float
def convert_to_float(x, default = True):
    if default:
        x = x.replace('.', '')
        x = x.replace(',', '.')
    else:
        x = x.replace(',', '')
    return float(x)


# Train and test set splitting
def rolling_split(data, train_size = 0.7):

    X = data.drop(columns=['return','datetime_clean','Price'])
    y = data['return']

    split_index = int(len(X) * train_size)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index+1:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index+1:]

    return X_train, X_test, y_train, y_test


# Function below will lag all variables one time, except for the return 3 times.
def shifter(data, dependent_only= False):

    if dependent_only == False:
        for col in data.columns:
            if col != "datetime_clean" and col != "Price" and col != "return" :
                data[f"{col}_lag_1"] = data[col].shift(1)
                data = data.drop([col], axis=1)
            if col == "return":
                for lag in range(1,4):
                    data[f"{col}_lag_{lag}"] = data[col].shift(lag)
    else:
        data = data[["return","datetime_clean","Price"]]
        for lag in range(1,4):
            data[f"return_lag_{lag}"] = data["return"].shift(lag)

    return data.dropna()


def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, verbose=True):
    """

    diebold-mariano test function according to the papers mentionned in the report and being inspired from
    a code source from a github page: "https://github.com/johntwk/Diebold-Mariano-Test". The principle
    author gave permission to modify the code according to its mentioned license.

    Returns:
        DM_stat: The diebold-mariano test statistic
        p_value: The relevant p_value of the test.

    """

    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst = []

    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

    # Length of lists (as real numbers)
    T = float(len(actual_lst))

    for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
        e1_lst.append((actual - p1) ** 2)
        e2_lst.append((actual - p2) ** 2)
    for e1, e2 in zip(e1_lst, e2_lst):
        d_lst.append(e1 - e2)

    # Mean of d
    mean_d = pd.Series(d_lst).mean()

    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(np.abs(k) + 1, N):
            autoCov += ((Xi[i - np.abs(k)]) - Xs) * (Xi[i] - Xs)

        return (1 / (T)) * autoCov

    gamma = []
    V_d = 0
    # for lag in tqdm(range(-(len(d_lst)-2),len(d_lst)-1), desc="Loading test"):

    S_t = int(0.75 * (len(d_lst) ** (1 / 3)))
    for lag in range(-S_t, S_t):
        # gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d))
        V_d += autocovariance(d_lst, len(d_lst), lag, mean_d)

    V_d = math.sqrt(V_d / T)

    DM_stat = mean_d / V_d
    p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

    DM_stat = round(DM_stat, 4)
    p_value = round(p_value, 4)

    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM=DM_stat, p_value=p_value)

    if verbose:
        return rt
    else:
        return DM_stat, p_value


def df_split(data, split_index=56480, ts=True):
    # split_index of 56480 corresponds to the 70% split of the total data
    # if data is used for time series model, use ts=True to return X_train and y_train together

    X = data.drop(columns=['return',
                           'Price'])

    y = data['return']

    split_index = split_index
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if ts:
        train_df = data.iloc[:split_index]
        return train_df, X_test, y_test

    else:
        return X_train, X_test, y_train, y_test


# Defining the expanding window function.
def rolling_window(data, train_end_idx=56480, app="ts", last_pred_idx=None):
    forecasts = []
    n = len(data)

    # Loop over the data with rolling window for each model class
    if app == "ts":
        for i in tqdm(range(train_end_idx, n), desc="Processing the rolling window"):
            train_data = data[i - 56480:i]  # Expanding window
            model = sm.tsa.AutoReg(train_data, lags=1).fit()
            forecast = model.predict(start=i + 1, end=i + 1)
            forecasts.append(forecast)

    elif app == "lm":
        X = sm.add_constant(data)
        try:
            X = data.drop(columns=['return'])
            X = X.drop(columns=["Price"])
        except:
            pass

        y = data['return']

        # Loop over the data with rolling window
        for i in tqdm(range(train_end_idx, n), desc="Processing the rolling window"):
            X_train = X.iloc[i - 56480:i]
            y_train = y.iloc[i - 56480:i]
            model = sm.OLS(y_train, X_train).fit()
            forecast = model.predict(X[i:i + 1])
            forecasts.append(forecast)

    elif app == "rf":
        X = data.drop(columns=['return', "Price"])
        y = data['return']
        last_pred_idx = n if last_pred_idx is None else last_pred_idx
        for i in tqdm(range(train_end_idx, last_pred_idx), desc="Processing the rolling window"):
            X_train = X.iloc[i - 56480:i]
            y_train = y.iloc[i - 56480:i]
            rf = RandomForestRegressor(random_state=42, n_estimators=25, n_jobs=-2)
            rf.fit(X_train, y_train)
            forecast = rf.predict(X[i:i + 1])
            forecasts.append(forecast)

    elif app == "dt":
        X = data.drop(columns=['return', "Price"])
        y = data['return']
        last_pred_idx = n if last_pred_idx is None else last_pred_idx
        for i in tqdm(range(train_end_idx, last_pred_idx), desc="Processing the rolling window"):
            X_train = X.iloc[i - 56480:i]
            y_train = y.iloc[i - 56480:i]

            dt = DecisionTreeRegressor(random_state=42)
            dt.fit(X_train, y_train)
            forecast = dt.predict(X[i:i + 1])
            forecasts.append(forecast)

    else:
        pass

    return forecasts

