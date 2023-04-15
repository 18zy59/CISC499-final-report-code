import pandas as pd
import requests
from io import StringIO, BytesIO
import zipfile

# download GDELT data
gdelt_url = "http://data.gdeltproject.org/events/20220101.export.CSV.zip"
response = requests.get(gdelt_url)

# unzip
zip_data = zipfile.ZipFile(BytesIO(response.content))
csv_data = zip_data.open(zip_data.namelist()[0])

# read data
gdelt_data = pd.read_csv(csv_data, sep='\t', header=None)
gdelt_data.columns = [
    'GlobalEventID', 'Day', 'MonthYear', 'Year', 'FractionDate', 'Actor1Code',
    'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode', 'Actor1EthnicCode',
    'Actor1Religion1Code', 'Actor1Religion2Code', 'Actor1Type1Code', 'Actor1Type2Code',
    'Actor1Type3Code', 'Actor2Code', 'Actor2Name', 'Actor2CountryCode',
    'Actor2KnownGroupCode', 'Actor2EthnicCode', 'Actor2Religion1Code',
    'Actor2Religion2Code', 'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
    'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode', 'QuadClass',
    'GoldsteinScale', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone',
    'Actor1Geo_Type', 'Actor1Geo_FullName', 'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code',
    'Actor1Geo_Lat', 'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type',
    'Actor2Geo_FullName', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code', 'Actor2Geo_Lat',
    'Actor2Geo_Long', 'Actor2Geo_FeatureID', 'ActionGeo_Type', 'ActionGeo_FullName',
    'ActionGeo_CountryCode', 'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long',
    'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
]


interactions = gdelt_data[['Actor1CountryCode', 'Actor2CountryCode']].dropna()

# save data to csv file
interactions.to_csv('C:/Users/Bill/Desktop/499country/diplomatic_relations.csv', index=False)
