from meteostat import Stations, Daily
from datetime import datetime, date,timedelta
import numpy as np
from pandas._libs.missing import NAType
import pandas as pd

stations = Stations()


SE_stations = stations.region('SE').fetch()
DE_stations = stations.region('DE').fetch()
FR_stations = stations.region('FR').fetch()

# the regional codes of france are outdated and are not the same as in
# mobility data. -> lets change them

FR_codes = pd.read_csv('FR_region_codes_new_old.csv')
FR_codes = FR_codes[['new','old']]

FR_stations = FR_stations.merge(FR_codes,left_on = 'region',right_on = 'old')
FR_stations['region'] = FR_stations['new']
FR_stations.drop(['new','old'],inplace = True,axis = 1)
FR_stations.drop_duplicates(inplace = True)


def fetch_weather_data_for_station(df,region):

    # Set time period
    start = datetime(2020, 2, 15)
    now = date.today()
    end = datetime(now.year,now.month,now.day,0,0)

    # Take away all stations that do not have produced hourly within the last month
    # -> these stations will not have daily data either
    df = df[df['hourly_end'] > end-timedelta(days=30)]

    # If there are no stations with weather data that covers the time span we want
    # -> this function returns an empty dataframe
    if df[df['region'] == region].name.count() == 0:
        return pd.DataFrame()

    loop = True
    while loop == True:
        # finds the closest weather station to the centroid of all the weather stations
        # in a given region
        # This try/except returns the data if there are no more stations available for searching

        try:
            station = df.loc[[df[df['region']==region][['latitude','longitude']].sub(df[df['region']==region][['latitude','longitude']].mean()).pow(2).sum(1).idxmin()]]
        except:            
            print('No more stations, returning: '+ station_name)
            return checkpoint


        # Chooses the value what to use for searching the weather station
        if type(station.iloc[0]['icao']) != NAType:
            station = stations.id('icao', station.iloc[0]['icao']).fetch()
        elif type(station.iloc[0]['wmo']) != NAType:
            station = stations.id('wmo', station.iloc[0]['wmo']).fetch()
        else:
            print( "Station id was not found" )
        

        # Get daily data
        data = Daily(station, start, end)
        data = data.fetch()
        station_name = station.iloc[0]['name']

        # Returns the data if there are not too many missing values
        if data.tavg.count() > 420 and data.prcp.count() > 300:
            print('returning data for station: ' + station_name)
            return data
        
        # In order to get the most complete possible data, every loop we compare
        # if the data searched for a new weather data is more complete than for the 
        # last searched
        try:
            if checkpoint.tavg.count() + checkpoint.prcp.count() < data.tavg.count() + data.prcp.count():
                print('New checkpoint: '+ station_name)
                checkpoint = data
            else:
                pass
        # this except catches the situation where the checkpoint gets checked the first time
        except:
            print('New checkpoint: '+ station_name)
            checkpoint = data

        print('Searching again..,\n Deleting station ' + station_name)
        # Deletes the row with the station for whom data was not found
        # -> makes the loop to search values for another weather station
        df = df[df['name'] != station_name]


# poista
def get_station(df,region):

    # finds the closest weather station to the centroid of all the weather stations
    # in a given region
    return df.loc[[df[df['region']==region][['latitude','longitude']].sub(df[df['region']==region][['latitude','longitude']].mean()).pow(2).sum(1).idxmin()]]
  
def get_all_weather_data(df):

    master_df = pd.DataFrame()
    for region in df.region.unique():
        buffer = fetch_weather_data_for_station(df,region)
        buffer['region'] = region
        master_df = pd.concat([master_df,buffer])
    
    return master_df


SE_weather_data = get_all_weather_data(SE_stations)
SE_weather_data.to_csv('SE_weather_data.csv',index_label = 'time')

FR_weather_data = get_all_weather_data(FR_stations)
FR_weather_data.to_csv('FR_weather_data.csv',index_label = 'time')

DE_weather_data = get_all_weather_data(DE_stations)
DE_weather_data.to_csv('DE_weather_data.csv',index_label = 'time')
