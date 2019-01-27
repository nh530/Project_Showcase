import pandas as pd

def hasPM25(df, column):
    '''
    Creates a bin with 2 labels.  Every 0 is labeled 0.  Any number greater or
    equal to 1 is assigned the label 1.  
    
    Parameters:
        df (pd.dataframe): The dataframe to be analyzed.
        column (String): name of a numerical column to bin.
    Return:
        Outputs a text file with the results to the work directory. 
    '''
    binLabels = [0, 1]
    maxN= df[column].max()
    binRange = [0, 1, maxN]
    df['binned '+column] = pd.cut(df[column], 
                    bins = binRange, right = False, labels = binLabels)
    return df

def cleanAirPollutionData(df):
    '''
    The purpose of this function is to clean the airPollutionData data frame
    by changing strings in the State and Country attributes, so it matches the 
    reference.  This function is specific to the airPollutionData data frame.
    
    Parameters:
        df (pd.dataframe): This must be the air pollution data set.  
    Return:
        clean (pd.dataframe): This is the cleaned air pollution data
    '''
    df[['State', 'County']] = df[['State','County']
                        ].apply(lambda x: x.str.replace(".","").str.strip())
    toChange = ["DeKalb", "DuPage", "Saint Clair", "McLean", "LaPorte", 
                "St John the Baptist", "Baltimore (City)", "Prince George's", 
                "Saint Louis", "DeSoto", "Saint Charles","Sainte Genevieve", 
                "McKinley", "McKenzie", "McClain", "Fond du Lac", 
                "Matanuska-Susitna","Yukon-Koyukuk"]
    newValue = ["De Kalb", "Du Page", "St Clair", "Mclean", "La Porte", 
                "St John The Baptist", "Baltimore", "Prince Georges",
                "St Louis", "De Soto", "St Charles", "Ste Genevieve", 
                "Mckinley", "Mckenzie", "Mcclain", "Fond Du Lac", 
                "Matanuska Susitna", "Yukon Koyukuk"]
    df['County'] = df['County'].replace(toChange, newValue)
    df['State'] = df['State'].replace('District Of Columbia',
                                      'District of Columbia')
    filt = df['State'].isin(['Tennessee', 'Virginia'])
    df.loc[filt, ['County']] = df.loc[filt, ['County']].replace('De Kalb', 
                                  'Dekalb').replace('Charles','Charles City')
    filt = (df['State'] != 'Country Of Mexico') & (
            df['State'] != 'Puerto Rico') & (df['State'] != 'Virgin Islands')
    clean = df[filt]
    return clean

def addFips(cleanedData):
    '''
    This function takes about 30 minutes to run.  It ends when the counter
    hits 5259.  Function's purpose is to add FIPS State and FIPS County to 
    cleanedData by matching county and state to countyReference. This function
    is not general and is specific to the air pollution data set used in this
    .py file.  
    
    Parameters:
        cleanedData (pd.DataFrame): Should be the air pollution data set.
    Return:
        withFips (pd.DataFrame): Returns the argument passed with FIPS added.
    '''
    withFips = cleanedData.copy()
    countyReference = pd.read_excel("https://www.schooldata.com/pdfs/" + \
                                "US_FIPS_Codes.xls", header=1)
    temp = countyReference['FIPS State'].unique()
    temp2 = countyReference['State'].unique()
    temp3 = pd.DataFrame({'State':temp2, 'FIPS State':temp})
    withFips['FIPS State'] = withFips['State'].map(temp3.set_index(
            "State")['FIPS State'])
    # This is a naive solution.  
    i=0
    countyRS = countyReference[['State','County Name','FIPS County']]
    cleanDS = withFips[['State','County']]
    for index, row in countyRS.iterrows():
        for index_d, row_d in cleanDS.iterrows():
            if row.State == row_d.State and row['County Name'] == row_d.County:
                withFips.loc[index_d, 'FIPS County'] = row['FIPS County']
                i+=1
                print(i)
    withFips['FIPS County'] = withFips['FIPS County']
    return withFips

def normalizeData(cdata):
    '''
    The air pollution AQI data must be normalized by the number of days 
    that the AQI values were collected.  Different counties have different
    number of days AQI were collected.  
    
    Parameter:
        cdata (pd.DataFrame): Should be the air pollution data set.  
    Return:
        norm (pd.DataFrame): Returns the dataframe passed, but now with 
        normalized columns.
    '''
    norm = cdata.copy()
    norm['Good Days_Norm'] = cdata['Good Days'] / cdata['Days with AQI']
    norm['Moderate Days_Norm'] = cdata['Moderate Days'] / cdata[
            'Days with AQI']
    norm['Unhealthy for Sensitive Groups Days_Norm'] = cdata[
            'Unhealthy for Sensitive Groups Days'] / cdata['Days with AQI']
    norm['Unhealthy Days_Norm'] = cdata['Unhealthy Days'] / cdata[
            'Days with AQI']
    norm['Very Unhealthy Days_Norm'] = cdata['Very Unhealthy Days'] / cdata[
            'Days with AQI']
    norm['Hazardous Days_Norm'] = cdata['Hazardous Days'] / cdata[
            'Days with AQI']
    norm['Days CO_Norm'] = cdata['Days CO'] / cdata['Days with AQI']
    norm['Days NO2_Norm'] = cdata['Days NO2'] / cdata['Days with AQI']
    norm['Days Ozone_Norm'] = cdata['Days Ozone'] / cdata['Days with AQI']
    norm['Days SO2_Norm'] = cdata['Days SO2'] / cdata['Days with AQI']
    norm['Days PM2.5_Norm'] = cdata['Days PM2.5'] / cdata['Days with AQI']
    norm['Days PM10_Norm'] = cdata['Days PM10'] / cdata['Days with AQI']
    return norm


def main():
    data2015 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/' + \
                           'annual_aqi_by_county_2015.zip')
    data2014 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/' + \
                           'annual_aqi_by_county_2014.zip')
    data2013 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/' + \
                           'annual_aqi_by_county_2013.zip')
    data2012 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/' + \
                           'annual_aqi_by_county_2012.zip')
    data2011 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/' + \
                           'annual_aqi_by_county_2011.zip')

    airPollutionData = pd.concat([data2015, data2014, data2013, 
                                  data2012, data2011])
    airPollutionData.reset_index(drop=True, inplace=True)
    cleanedData = cleanAirPollutionData(airPollutionData)
    cleanedData = hasPM25(airPollutionData, 'Days PM2.5')
    cdata = addFips(cleanedData)
    cdata = normalizeData(cdata)
    
    # Outliers were determined by looking at boxplots of each variable.  
    clean_no_outlier = cdata[~(cdata['Max AQI'] >=500)] 
    clean_no_outlier['FIPS State'] = clean_no_outlier['FIPS State'].apply(
            '="{}"'.format)
    clean_no_outlier['FIPS County'] = clean_no_outlier['FIPS County'].apply(
            '="{}"'.format)
    clean_no_outlier.to_csv('cleanAirData.csv', index=False)
    
main()

