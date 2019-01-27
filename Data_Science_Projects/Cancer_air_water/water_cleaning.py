import pandas as pd
import urllib
import urllib.request
import json


def main():
    url = "https://ephtracking.cdc.gov:443/apigateway/api/v1/" + \
            "getCoreHolder/441/2/ALL/ALL/2011,2012,2013,2013,2014,2015/0/0"
    # Reading the json as a dict
    with urllib.request.urlopen(url) as json_data:
        data = json.load(json_data)
    dataset = pd.DataFrame.from_dict(data['pmTableResultWithCWS'])  
    # Datatype of dataValue is object, change it into numeric
    dataset['dataValue'] = pd.to_numeric(data['dataValue'], errors='coerce')
    
    # 1. get the means of all values for each county using "groupby" method
    result = data.groupby(['title','year','geoId'])['dataValue'].mean() 
    result = result.to_frame().reset_index()
    result.columns = ['Location','Year','GeoId','Value']
    # 2. split the location column into two column("county","state")
    result[['County','State']] = result['Location'].str.split(',', n=1, 
                                                              expand=True)
    del result['Location']
    result['Quality'].dropna(inplace=True)
    result.to_csv("water_clean.csv",index=False)
