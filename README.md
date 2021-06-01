# QDC
Code for Lassi's, David's and Shiyuan's QDC

## What the code does

Cleans the data used (clean.py)
Fetches weather data


## Howto initialize and run

1.) Run weather.py to get the newest weather data from meteostat's API

2.) Download data from the following sources:

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
