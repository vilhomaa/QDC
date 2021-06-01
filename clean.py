import pandas as pd


##########################
## DATA CLEANING SCRIPT ##
##########################

## The idea of the script is to prepare the data for the Analysis

## Dependencies:
# weather data generated with weather.py
# data of the amount of cases (links below)
# country related data with names of areas and their respective codes
#   - country related files are made manually from various sources


## DATA SOURCES:
"""
Mobility data: https://www.google.com/covid19/mobility/


FRANCE :
CASES : https://www.data.gouv.fr/en/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/#_
AFTER DOWNLOADING FRANCE'S DATA, RENAME IT TO 'france_raw.csv'
Regions : https://www.data.gouv.fr/en/datasets/regions-departements-villes-et-villages-de-france-et-doutre-mer/

GERMANY:
Cases: https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/6d78eb3b86ad4466a8e264aa2e32a2e4_0/data?orderBy=BundeslandId

SWEDEN:
Cases: https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/statistik-och-analyser/bekraftade-fall-i-sverige/
"""


## STRUCTURE:
#       1. IMPORT COUNTRY-SPECIFIC DATA
#           A. Combine data with the names and populations of 
#               the country-specific regions
#           B. Generate the average cases per 100k inhabitants
#               during the last 7 days
#           C. Other adjustements
#               - Adding the Neurohm index
#
#       2. IMPORT WEATHER-RELATED DATA
#           - Generated with weather.py
#       3. ALL DATA SETS



## DATA FOR GERMANY

RKI = pd.read_csv('RKI_history.csv',index_col= False)

# Takes all the regions, discards all subregions of the german lands
DE = RKI.loc[RKI['AdmUnitId'] == RKI['BundeslandId']]

# join the names of the regions
lander = pd.read_csv('lander.csv',sep=',')
DE_merged = DE.merge(lander,right_on='id',left_on='BundeslandId')

# date to date and sorting
DE_merged['Datum'] = pd.to_datetime(DE_merged['Datum'],format='%Y/%m/%d %H:%M:%S+%f')
DE_merged.sort_values(['BundeslandId','Datum'],inplace = True)

# generate inzidenztal
DE_merged['cases_past_7_days'] = DE_merged.groupby('BundeslandId')['AnzFallVortag'].transform(lambda x: x.rolling(7, 1).sum())

# generate cases/100k inhabitants
DE_merged['cases_per_100k'] = DE_merged['cases_past_7_days']/(DE_merged['pop'].str.replace(" ","").astype(int)/100)

# Converting population numbers to be in the same format as all the other countries'
DE_merged['pop'] = pd.to_numeric(DE_merged['pop'].str.replace(' ',''))
DE_merged['pop'] = DE_merged['pop'] * 1000

## SWEDEN DATA

SE_raw = pd.read_excel('Folkhalsomyndigheten_Covid19.xlsx',sheet_name="Antal per dag region")

SE_raw = SE_raw.set_index('Statistikdatum').stack().reset_index()
SE_raw.columns = ['Datum','county','cases']

SE_regional_pop = pd.read_csv('SE_region_population.csv')
# drops extra columns generated for some reason
SE_regional_pop = SE_regional_pop[['county','pop','code']]

# Merge (will drop values for the whole of sweden)
SE_merged = SE_raw.merge(SE_regional_pop,on = 'county')
SE_merged.sort_values(['county','Datum'],inplace = True)

# calculate 7 day incidence rate per 100k inhabitants
SE_merged['cases_past_7_days'] = SE_merged.groupby('county')['cases'].transform(lambda x: x.rolling(7, 1).sum())

# generate cases/100k inhabitants
SE_merged['cases_per_100k'] = SE_merged['cases_past_7_days']/(SE_merged['pop']/100000)


## FRANCE DATA
# REMINDER: RENAME THE CSV FILE GOTTEN FROM THE LINK TO FRANCE'S DATA TO:
# 'france_raw.csv'
FR_raw = pd.read_csv('france_raw.csv',sep=';')

# add region names
FR_regions = pd.read_csv('fr_regions.csv')

FR_regions['code_no'] = FR_regions['code_no'].replace('COM','999')
FR_regions['code_no'] = FR_regions['code_no'].astype(int)
FR_merged = FR_raw.merge(FR_regions[['code_no','name','code']],left_on='reg', right_on='code_no')

FR_merged = FR_merged.drop(['code_no'],axis=1)
FR_merged = FR_merged.rename(
    columns={"name": "sub_region_1"}
)

# keep only observations including all age groups
FR_merged = FR_merged[FR_merged['cl_age90'] == 0]

# date to date and sorting
FR_merged['jour'] = pd.to_datetime(FR_merged['jour'],format='%Y-%m-%d')
FR_merged.sort_values(['reg','jour'],inplace = True)

# generate inzidenzzahl
FR_merged['cases_past_7_days'] = FR_merged.groupby('reg')['P'].transform(lambda x: x.rolling(7, 1).sum())

# generate cases/100k inhabitants
FR_merged['cases_per_100k'] = FR_merged['cases_past_7_days']/(FR_merged['pop']/100000)


## ADDING THE NEUROHM INDEX 
# these are the regression coefficients from neurohm.py
"""
DE             0.2711
FR             0.5944
SE             0.3419
"""
# We will divide these by two, to get a standardized value 
# The maximum value is two

DE_merged['neurohm_idx'] = 0.2711/2
FR_merged['neurohm_idx'] = 0.5944/2
SE_merged['neurohm_idx'] = 0.3419/2


## MERGING ALL COUNTRY DATA TO 1 DF
cols_from_country_data = [
    'time',
    'cases',
    'population',
    'region',
    'cases_past_7_days',
    'cases_per_100k', 
    'neurohm_idx'
]

# Merging all the data with the amount of cases to one df
# first - rename the columns to have same names
FR_merged = FR_merged.rename(columns={
    'jour' : 'time', 
    'P' : 'cases',
    'pop' : 'population',
    'code' : 'region'
})
DE_merged = DE_merged.rename(columns={
    'Datum' : 'time', 
    'AnzFallVortag' : 'cases',
    'pop' : 'population',
    'code' : 'region'
})
SE_merged = SE_merged.rename(columns={
    'Datum' : 'time', 
    'pop' : 'population',
    'code' : 'region'
})

FR_merged = FR_merged[cols_from_country_data]
DE_merged = DE_merged[cols_from_country_data]
SE_merged = SE_merged[cols_from_country_data]

DE_merged['country_region_code'] = 'DE'
FR_merged['country_region_code'] = 'FR'
SE_merged['country_region_code'] = 'SE'

country_data = pd.concat([FR_merged,DE_merged,SE_merged])


## IMPORT WEATHER DATA
DE_weather = pd.read_csv('DE_weather_data.csv')
FR_weather = pd.read_csv('FR_weather_data.csv')
SE_weather = pd.read_csv('SE_weather_data.csv')

DE_weather['country_region_code'] = 'DE'
FR_weather['country_region_code'] = 'FR'
SE_weather['country_region_code'] = 'SE'

weather_data = pd.concat([DE_weather,FR_weather,SE_weather])

# Only take interesting columns
weather_columns_of_interest = [
    'time', # date
    'tavg', # avg temp of the day
    'prcp', # Precipitation
    'wspd', # windspeed
    'region', # region code
    'country_region_code'
]

weather_data = weather_data[weather_columns_of_interest]



# COMBINING THREE DATA SOURCES

mobility = pd.read_csv('Global_Mobility_Report.csv',index_col = False)
countries_to_keep = ['DE','SE','FR']
mobility = mobility[mobility['country_region_code'].isin(countries_to_keep)]

# Making the region code for joining
mobility['region'] = mobility['iso_3166_2_code'].str[3:]

columns_to_keep_mobility = [
    'country_region_code',
    'sub_region_1',
    'region',
    'date',
    'retail_and_recreation_percent_change_from_baseline',
    'grocery_and_pharmacy_percent_change_from_baseline',
    'parks_percent_change_from_baseline',
    'transit_stations_percent_change_from_baseline',
    'workplaces_percent_change_from_baseline',
    'residential_percent_change_from_baseline'
]
mobility = mobility[columns_to_keep_mobility]

# converting datatypes to be similar so that they can be joined

mobility['time'] = pd.to_datetime(mobility['date'])
mobility.drop(columns = ['date'],inplace = True)

country_and_mobility = pd.merge(mobility,country_data,on= ['time','region','country_region_code'])


weather_data['time'] = pd.to_datetime(weather_data['time'])
final_df = pd.merge(country_and_mobility,weather_data,on= ['time','region','country_region_code'])

# Checking that all the regions are included in the newly merged df
assert len(list(set(final_df.region.unique()) - set(mobility.region.unique()))) == 0 , print('regions lost from country_data')
assert len(list(set(final_df.region.unique()) - set(weather_data.region.unique()))) == 0 ,print('regions lost from weather data')


final_df.to_csv('final_data.csv')

