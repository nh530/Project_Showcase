import pandas as pd
import io
import requests
import numpy as np

def load_data(url, category, groupname, cancer_data):
    '''
    The purpose of this function is to load the cancer data set from 
    cancer.gov.  This function is not generalizable to every csv file on the 
    internet.  
    
    Paramters: 
        url (String): This is the url for the location of the csv file to be
        retrieved.
        category (String): Represents the category of the data set. 
        Group (String): Represents the sub-category of the data set.
        cancer_data (pd.DataFrame): This is the dataframe object instance to
        house the cancer data set.  
    Return: 
        cancer_data (pd.DataFrame): This is the cancer_data object that was 
        passed as an argument, but now it has the new data that was pulled from
        the url appended to it.  
    '''
    s = requests.get(url).content
    sub_data = pd.read_csv(io.StringIO(s.decode('windows-1252')), 
                           skiprows=8, skipfooter=27, engine='python', 
                           error_bad_lines=False)   
    sub_data['Category'] = category
    sub_data['Group'] = groupname
    return (cancer_data.append(sub_data, ignore_index = True))

def clean(df, sym):
    '''
    This function performs data cleaning steps on the cancer data set.   
    
    Parameters:
        df (pd.DataFrame): Cancer dataframe object instance.  
        sym (list object): A list of strings to be removed from the cancer
        data set.  The strings must correspond to values in the cancer object
        instance.  
    Return:
        temp (pd.DataFrame): A clean cancer data set.  
    '''
    temp = df.copy()
    temp = temp.drop("Met Healthy People Objective of ***?", axis=1)
    for i in sym:
        temp.replace(i, np.nan, inplace=True)
    # This indexing operation drops all rows where FIPS equals 0.  
    temp = temp[temp[' FIPS'] != 0.0]
    # Delete the space before FIPS variable name.
    temp.rename(columns={' FIPS': 'FIPS'}, inplace=True)
    # Replace fewer than 3 Average Annual Count into 3.
    temp = temp.replace('3 or fewer','3')
    # change Recent 5-Year Trend (‡) in Incidence Rates & Average Annual Count 
    # into numeric variable
    temp['Recent 5-Year Trend (‡) in Incidence Rates'] = temp[
            'Recent 5-Year Trend (‡) in Incidence Rates'].astype('float')
    temp['Average Annual Count'] = temp['Average Annual Count'
               ].astype('float')
    # delete '#' 
    temp['Age-Adjusted Incidence Rate(†) - cases per 100,000'
    ] = temp['Age-Adjusted Incidence Rate(†) - cases per 100,000'
    ].str.replace('#',"")
    #split county, state, SEER, NPCR into separate columns
    temp['County'], temp['State'] = temp['County'].str.split(
            ', ', 1).str
    temp['State'], temp['SEER'] = temp['State'].str.split(
            '(', 1).str
    temp['SEER'], temp['NPCR'] = temp['SEER'].str.split(
            ',', 1).str
    temp['NPCR'] = temp['NPCR'].str.replace(")","")          
    # add leading zeros to FIPS < 5 digits
    temp['FIPS'] = temp['FIPS'].astype(str).replace("000nan", "")
    temp['FIPS'], na = temp['FIPS'].str.split('.', 1).str
    del na
    temp['FIPS'] = temp['FIPS'].apply(lambda x: '{0:0>5}'.format(
            x))     
    return temp     

def remOutlier(df):
    '''
    This function removes outliers from the cancer data set.  Outliers were
    detected using histograms.  
    
    Parameter:
        df (pd.DataFrame): Cancer dataframe instance.
    Return:
        temp (pd.DataFrame):  Cancer data set with outliers removed.  
    '''
    temp = df.copy()
    temp.loc[temp['Average Annual Count'] >2500]
    return temp

def main():
    cancer_data = pd.DataFrame()
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                            '=001&race=00&sex=0&age=001&type=incd&' + \
                            'sortVariableName=rate&sortOrder=desc&output=1',
                 'All', 'Total', cancer_data)

    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                        'incidencerates/index.php?stateFIPS=99&cancer=' + \
                              '047&race=00&sex=0&age=001&type=incd&sort' + \
                               'VariableName=rate&sortOrder=desc&output=1',
                    'Cancer Type', 'Lung Cancer', cancer_data)
    
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                               '=001&race=06&sex=0&age=001&type=incd&sort' + \
                               'VariableName=rate&sortOrder=desc&output=1',
                 'Race/Ethnicity', 'White Hispanic', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                            '=001&race=07&sex=0&age=001&type=incd&sortVar' + \
                            'iableName=rate&sortOrder=desc&output=1',
                 'Race/Ethnicity', 'White Non-Hispanic', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                            '=001&race=02&sex=0&age=001&type=incd&sortVar' + \
                               'iableName=rate&sortOrder=desc&output=1',
                 'Race/Ethnicity', 'Black (includes Hispanic)', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                        'incidencerates/index.php?stateFIPS=99&cancer=' + \
                        '001&race=04&sex=0&age=001&type=incd&sortVar' + \
                            'iableName=rate&sortOrder=desc&output=1',
            'Race/Ethnicity', 'Asian or Pacific Islander (includes Hispanic)',
                 cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                            '=001&race=00&sex=1&age=001&type=incd&sortVar' + \
                               'iableName=rate&sortOrder=desc&output=1',
                               'Sex', 'Males', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                              '=001&race=00&sex=2&age=001&type=incd&sort' + \
                               'VariableName=rate&sortOrder=desc&output=1',
                               'Sex', 'Females', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                            '=001&race=00&sex=0&age=009&type=incd&sortVar' + \
                               'iableName=rate&sortOrder=desc&output=1',
                               'Age', '<50', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                            'incidencerates/index.php?stateFIPS=99&cancer' + \
                        '=001&race=00&sex=0&age=136&type=incd&sortVar' + \
                            'iableName=rate&sortOrder=desc&output=1',
                               'Age', '50+', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                        'incidencerates/index.php?stateFIPS=99&cancer=' + \
                               '001&race=00&sex=0&age=006&type=incd&sort' + \
                               'VariableName=rate&sortOrder=desc&output=1',
                 'Age', '<65', cancer_data)
    cancer_data = load_data('https://www.statecancerprofiles.cancer.gov/' + \
                        'incidencerates/index.php?stateFIPS=99&cancer' + \
                             '=001&race=00&sex=0&age=157&type=incd&sort' + \
                               'VariableName=rate&sortOrder=desc&output=1',
                 'Age', '65+', cancer_data)
    # delete Met Healthy People column
    cancerCopy = cancer_data.copy()
    sym = ['*', '* ', ' *', '**', ' **', '** ', '¶', '¶ ', '¶¶', '¶¶ ',' ¶¶',
           ' ¶','§§§', ' §§§', '§§§ ', '§§', '§§ ', 
           ' §§', '&sect;&sect;&sect;']   
    cancerCopy = clean(cancerCopy, sym)
    cancerCopy = remOutlier(cancerCopy)
    cancerCopy.to_csv('cleaned_cancer.csv', encoding='utf-8', index = False)

main()