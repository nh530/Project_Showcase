import urllib
import urllib.request
import pandas as pd
import json
import io
import requests

### create summary stats 
def checkType(dataset, filename):
    file1 = open(filename,'a')
    file1.write("\nOverall review of dataset:\n")
    file1.write(dataset.describe().transpose().to_string())
    file1.close()
    
# function to write size/shape results to a .txt
def dataInfo(dataset, filename):
    file1 = open(filename,'a')
    file1.write('\n\nSize of the dataframe is ' + str(dataset.shape) +'.\n\n')
    file1.write('There are ' + str(dataset.size) + ' elements in this dataset.\n\n')
    file1.write('Data types of each columns are\n ' + dataset.dtypes.to_string() +'\n\n')
    file1.close()

##count of null rows, null values in columns
##isnull() returns a dataframe that contains true or false depending on if null or not.
##values() returns a 2-d array that contains only the values from the dataframe.
##ravel() turns n-dimensions into 1-dimension by flattening the array.
##then sum() is used to sum up each element.  False is 0 and True is 1.
def nullCount(dataset, filename):
    file1 = open(filename,'a')
    nulls = dataset.isnull().values.ravel().sum()
    file1.write('\nThe total number of rows with null values is: \n' + str(nulls) +'.\n\n')
    file1.write("Count of missing value for each column:\n")
    file1.write(pd.isna(dataset).sum().to_string())
    file1.close()

# get unique value for each variable
def getUnivalue(dataset, filename):
      ##list of column names
      names = list(dataset)
      for i in range(0,len(names)-1):
            var = names[i]
            unique_values = dataset[var].unique()
            file1 = open(filename,'a')
            file1.write('\n\nThe unique values for ' + ' " '+ str(var) + '"' 
                    + ' are ' + str(unique_values) + '.\n\n')
            file1.close()
        
def genFnsWrapper(dataset, filename):
    checkType(dataset, filename)
    dataInfo(dataset, filename)
    nullCount(dataset, filename)
    getUnivalue(dataset, filename)
    
##break statement is used to exit out of the nearest for loop
##The purpose of this loop is to determine invalid States and Counties by comparing the 
##States and Counties of interset with a reference.
##The output has repitition because there are multiple state,county entries within some keys.
##These aren't duplicates because the same state,county is used across different years.
def stateCountyChecker(a,b):
    countCounty = 0
    countState = 0
    
    for key in a:
          for value in a[key]:
                try:
                      if value not in b[key]:
                            print("The following state|county is a messy data entry: ", key,'|', value)
                            countCounty += 1
                except KeyError:
                      print(key,"is not a valid State or the entry is messy")
                      countState += 1
                      break
    file1 = open('Air_data_analysis.txt','a')
    file1.write('\n\nThe number of messy States is: ' + str(countState))
    file1.write('\n\nThe number of messy Counties is: ' + str(countCounty))
    file1.close()              


##Leap years have 366 days.  2012 and 2016 are leap years
##The purpose of this loop is to run through each column and determine the extent of messy data
def numericColumnChecker(allPollutionData,columns):
    for i in columns:
          c = allPollutionData[i].value_counts().sort_index()
          if i == 'Year':
                file1 = open('Air_data_analysis.txt','a')
                file1.write('\n\nExpecting years to be between 2011 and 2017')
                file1.close()   
                indexLength = len(c)-1
                if c.index[0] < 2011 or c.index[indexLength] > 2017:
                      invalidEntries = c[(c.index < 2011) | (c.index > 2017)].sum()
                      file1 = open('Air_data_analysis.txt','a')
                      file1.write('\n\nThere are invalid entries in years column')
                      file1.write('\nThe number of invalid entries is: '+str(invalidEntries))
                      file1.close()
                else:
                    file1 = open('Air_data_analysis.txt','a')
                    file1.write('\n\nNo invalid entries for: ' +str(i))
                    file1.close()                     
          elif (i == 'Days with AQI' or i == 'Good Days' or i == 'Moderate Days' or i == 'Days CO'
                or i == 'Days NO2' or i == 'Days Ozone' or i == 'Days SO2' or i == 'Days PM2.5'
                or i == 'Days PM10'):
                  file1 = open('Air_data_analysis.txt','a')
                  file1.write('\n\nExpecting entries for ' + str(i) + ' column to between 0 and 366')
                  file1.close()
                  indexLength = len(c)-1               
                  if c.index[0] < 0 or c.index[indexLength] > 366:
                      invalidEntries = c[(c.index < 0) | (c.index > 366)].sum()
                      file1 = open('Air_data_analysis.txt','a')
                      file1.write('\n\nThere are invalid intries in '+ str(i))
                      file1.write('\nThe number of invalid entries is: '+ str(invalidEntries))
                      file1.close()
                  else:
                      file1 = open('Air_data_analysis.txt','a')
                      file1.write('\n\nNo invalid entries for' +str(i))
                      file1.close()            
          elif (i == 'Max AQI' or i == '90th Percentile AQI' or i == 'Median AQI'):
                file1 = open('Air_data_analysis.txt','a')
                file1.write('\n\nExpecting entries for' +str(i)+' column to be greater than 0')
                file1.close() 
                if c.index[0] < 0:
                      invalidEntries = c[c.index < 0].sum()
                      file1 = open('Air_data_analysis.txt','a')
                      file1.write('\n\nThere are invalid intries in'+str(i))
                      file1.write('\nThe number of invalid entries is: '+ str(invalidEntries))
                      file1.close()    
                else:
                      file1 = open('Air_data_analysis.txt','a')
                      file1.write('\n\nNo invalid entries for '+ str(i))
                      file1.close()  
                     

##creating a bin for the Days PM2.5 variable.
##0 is when there is zero days with PM2.5 and 1 is when there are days with PM2.5
def hasPM25(airPollutionData):
    binLabels = [0,1]
    binRange = [0,1,367]
    airPollutionData['hasPM2.5'] = pd.cut(airPollutionData['Days PM2.5'],bins = binRange, 
                    right = False, labels = binLabels)
    return airPollutionData


#######
### Water Data features
#######
def waterClean(data):
    ### Datatype of dataValue is object, change it into numericals
    data['dataValue'] = pd.to_numeric(data['dataValue'], errors='coerce')
    ### Checking missing values/typos/outliers in datasets  
    with open ('waterCheck.txt','w') as wc:
#        wc.write("\nHave a general understanding of the dataset:\n")
#        wc.write(data.info().to_string())
        wc.write("\nvalue_counts() function: check is there any error value in 'display' colomn\n")
        wc.write(data['display'].value_counts().to_string())
        wc.write("\nvalue_counts() function:check is there any error value in 'dataValue' colomn\n")
        wc.write(data['dataValue'].value_counts().to_string())
        wc.write("\nvalue_counts() function:check is there any error value in 'title' colomn \n")
        wc.write(data['title'].value_counts().to_string()) 
    
    ### Datatype of dataValue is object, change it into numericals
    data['dataValue'] = pd.to_numeric(data['dataValue'], errors='coerce')
    
    
    ### 1. get the means of all values for each county using "groupby" method
    result = data.groupby(['title'])['dataValue'].mean()
    ### change "series" into "dataframe" datatype
    with open ('waterCheck.txt','a') as wc:
        wc.write("\nThe mean value of arsenic concerntration for each county\n")
        wc.write(result.to_string())
        
    ### change "series" into "dataframe" datatype    
    result = result.to_frame().reset_index()
    result.columns = ['Location','Value']
    
    ### 2. split the location column into two column("county","state"), expand=True to add these two column into dataframe
    result[['county','state']] =result['Location'].str.split(',', n=1, expand=True)
    ### remove the location column
    del result['Location']
    print (result.describe())
    ### the max value is 31.837500, which is lower than 50
    
    
    ### 3. binning the mean into three categories
    ### According to documents in its original website:
    ### level < 1 means non-dect arsenic
    ### level in (1-10) means less than MCL == "no harm"
    ### level in (10-50) means "harmful"
    bins = [-1,1,10,50]
    labels=['Non Detect','Less than or equal MCL','More than MCL' ]
    ### 4. remove non detect value of water quality 
   
    result['Quality']=pd.cut(result['Value'],bins,labels=labels)
    result = result[(result['Quality'] != 'Non Detect')]
    with open ('waterCheck.txt','a') as wc:
        wc.write("\nThe cleaned dataset and new dataset features\n")
        wc.write(result.to_string())
    
    return result
    
def main():
    ### Water Data
    url='https://ephtracking.cdc.gov:443/apigateway/api/v1/getCoreHolder/441/2/ALL/ALL/2015/0/0'
    waterResp = urllib.request.urlopen(url)
    waterRawdata = json.loads(waterResp.read().decode())
    # read json into dataframe, "dict" format, cannot read dict directly
    waterDF=pd.DataFrame.from_dict(waterRawdata['pmTableResultWithCWS'])
    del waterDF["rollover"]
    #run general analysis
    genFnsWrapper(waterDF, 'Water_data_analysis.txt')
    #### output into .csv file, optional
    waterDF.to_csv('uncleaned_waterQuality.csv', sep=',', encoding='utf-8')
    ### use cleaning data function
    waterResult = waterClean(waterDF)

    ### Cancer Data
    url = 'https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1'
    s = requests.get(url).content
    cancer_data = pd.read_csv(io.StringIO(s.decode('windows-1252')), skiprows=8, skipfooter=27, engine='python')   
    cancer_data.to_csv('uncleaned_cancer.csv', sep=',', encoding='utf-8')
    genFnsWrapper(cancer_data, 'Cancer_data_analysis.txt')
    
    ### Air quality Data
    data2017 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2017.zip')
    data2016 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2016.zip')
    data2015 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2015.zip')
    data2014 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2014.zip')
    data2013 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2013.zip')
    data2012 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2012.zip')
    data2011 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2011.zip')
    countyReference = pd.read_excel("https://www.schooldata.com/pdfs/US_FIPS_Codes.xls")


    ##fixing the header for CountyReference
    new_header = countyReference.iloc[0]
    countyReference = countyReference[1:]
    countyReference.columns = new_header
    ##creating a single dataframe for air quality data and exporting to a csv file
    allPollutionData = pd.concat([data2017, data2016, data2015, data2014, data2013, data2012, data2011])
    #general analysis
    genFnsWrapper(allPollutionData, 'Air_data_analysis.txt')
    allPollutionData.to_csv('All_Pollution_Data.csv')
    dataOnlyStateCounty = allPollutionData.loc[:,['State','County']]
    referenceStateCounty = countyReference.loc[:,['State','County Name']]
    ##groupby('State') is grouping the dataframe by the unique values in the State column.
    ##The values in Country column are mapped to each unique value from State column.
    ##Tthen index for County column and turn the values that are mapped to the State column into 
    ##each distinct list groupings.
    ##Then turn each unique grouping to a dict
    a = dataOnlyStateCounty.groupby('State')['County'].apply(list).to_dict()
    b = referenceStateCounty.groupby('State')['County Name'].apply(list).to_dict()
    columns = ['Year','Days with AQI','Good Days','Moderate Days','Max AQI','90th Percentile AQI',
               'Median AQI','Days CO','Days NO2','Days Ozone',
               'Days SO2','Days PM2.5','Days PM10']
    
    stateCountyChecker(a,b)
    numericColumnChecker(allPollutionData, columns)
    
    allPollutionData = hasPM25(allPollutionData)
    with open ('airDataWithNewFeature.txt','a') as f:
          f.write("\nThe cleaned dataset and new dataset features\n")
          f.write(allPollutionData.to_string())
    

main()





