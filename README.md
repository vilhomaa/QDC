# QTEM Data Challenge
Code for Lassi's, David's and Shiyuan's QDC

## What the code does

Fetches weather data (weather.py
Cleans the data used (clean.py)

Runs all the analyses used in our report and generates the graphs as well. 

(Runtime on lassis laptop circa 3h, because it runs the analysis for forecasting 1 to 30 days.
if you are interested in a shorter runtime, limit the amount of days by changing the list day_amounts to include fewer days)


## Howto initialize and run

1.) Run weather.py to get the newest weather data from meteostat's API
        - make sure you have the FR_region_old_new.csv in your working directory

2.) Download data from the following sources and add them to your working directory:

"""
Mobility data: https://www.google.com/covid19/mobility/


FRANCE :
CASES : https://www.data.gouv.fr/en/datasets/donnees-relatives-aux-resultats-des-tests-virologiques-covid-19/#_
AFTER DOWNLOADING FRANCE'S DATA, RENAME IT TO 'france_raw.csv'

Regions : https://www.data.gouv.fr/en/datasets/regions-departements-villes-et-villages-de-france-et-doutre-mer/
RENAME THE REGION DATA TO 'fr_regions.csv'

GERMANY:
Cases: https://npgeo-corona-npgeo-de.hub.arcgis.com/datasets/6d78eb3b86ad4466a8e264aa2e32a2e4_0/data?orderBy=BundeslandId

SWEDEN:
Cases: https://www.folkhalsomyndigheten.se/smittskydd-beredskap/utbrott/aktuella-utbrott/covid-19/statistik-och-analyser/bekraftade-fall-i-sverige/
"""

3.) Run clean.py

4.) Run analysis.py
