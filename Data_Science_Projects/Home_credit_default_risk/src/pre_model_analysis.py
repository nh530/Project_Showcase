import pandas as pd 
import sys
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 300)

def missing_value_calculator(dataset):
    # which columns have missing values
    indexer = dataset.isnull().sum() > 0
    
    # Raw counts of missing values
    raw = dataset.isnull().sum()[indexer].sort_values(ascending=False)
    
    # percentage of missing values
    percentage = ((dataset.isnull().sum()/dataset.shape[0])
                    *100)[indexer].sort_values(ascending=False)

    # combine the two dataframes into 1 output
    output = pd.concat([raw, percentage], axis=1, ignore_index=False, 
                       keys=['# Missing', 'Percent Missing'])
    return output


def iterate_list(list):
    string = ''
    for element in list:
        string += str(element)
        string += '\n'
    return string


def categorical_field_writer(dataframe, file):
    cat_writer = open('../output/' + file + '_' + 'analysis.txt', 'a')
    cat_columns = dataframe.select_dtypes('object').columns
    for column in cat_columns:
        cat_str_obj = iterate_list(dataframe.loc[:, column].unique())
        if len(cat_str_obj) < 100:
            cat_writer.write('Unique Values for ' + column + ' Column:\n')
            cat_writer.write(cat_str_obj)
            cat_writer.write('\n\n\n')
    cat_writer.close()

    
def analysis_writer(app_train, dependent_variable, file):
    writer = open('../output/' + file + '_' + 'analysis.txt', 'a')
    writer.write('DataFrame Shape:\n(# rows, # columns) = ')
    writer.write(str(app_train.shape))
    writer.write('\n\n\n')
    baseline = app_train.dropna()
    writer.write('Dataframe Shape After Dropping Every row with nan:\n')
    writer.write(str(baseline.shape))
    writer.write('\n\n\n')
    writer.write('Classification Dependent Variable Distribution:\n')
    writer.write(app_train[dependent_variable].value_counts().to_string())
    writer.write('\n\n\n')
    writer.write('Columns With Missing Values:\n')
    missing_values_df = missing_value_calculator(app_train)
    writer.write(missing_values_df.to_string())
    writer.write('\n\n\n')
    writer.write('Column Data Type Distribution:\n')
    writer.write(app_train.dtypes.value_counts().to_string())
    writer.write('\n\n\n')
    writer.write('Descriptive Statistics On float64 Data Type Columns:\n')
    writer.write(str(app_train.select_dtypes('float64').apply(
                pd.Series.describe, axis=0)))
    writer.write('\n\n\n')
    writer.write('Descriptive Statistics On int64 Data Type Columns:\n')
    writer.write(str(app_train.select_dtypes('int64').apply(
                pd.Series.describe, axis=0)))
    writer.write('\n\n\n')
    writer.write('List Of object Data Type Columns:\n')
    str_obj = iterate_list(app_train.select_dtypes('object').columns)
    writer.write(str_obj)
    writer.write('\n\n\n')
    writer.write('Number Of Unique Values For Categorical Variables:\n')
    writer.write(app_train.select_dtypes('object').apply(
                pd.Series.nunique, axis=0).to_string())
    writer.write('\n\n\n')
    writer.close()
    
    categorical_field_writer(app_train, file)
    
    
def main():
    filename = sys.argv[1]
    dpt_var = sys.argv[2]
    train = pd.read_csv('../input/' + filename)
    
    analysis_writer(train, dpt_var, filename)
    
if __name__ == '__main__':
    main()

