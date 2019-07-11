import time
import csv
import gc

import lightgbm as lgb
from numpy import log
from numpy import sqrt
from numpy import mean
from numpy import square
from numpy import log1p
from numpy import expm1
from numpy import array

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV

'''
NOTES
    
# If you plot the distribution of raw price. You can get the highly skewed distribution with 
# a long tail on the right-hand side (since the maximal price has no limits). 
# In that case, you can expect that there are many outliers far away from 
# the peak and this distribution deviates from Gaussian distribution greatly. 
# This might not be good for the least squared optimizer. On the other hand, 
# when transforming y to log scale, it alleviates the long tail (become 
# shorter one) and the deviation from normality. It might be better for 
# the least squared optimizer to find a good solution. 
'''

def preprocessing(dataset):
    dataset = dataset.copy()
    start_time = time.time()    
    
    # NUM_BRANDS = train.brand_name.unique()
    # There is a total of 6311 brands
    NUM_BRANDS = 5054 # Removing all brands with 1 count.  
    '''
    NOTES
    
    We want there to be some variation in the fields.  If field only has frequency 
    of 1, then there is no variation and it wont be helpful in prediction.  
    
    '''
#    MAX_FEATURES_ITEM_DESCRIPTION = 2
    NAME_MIN_DF = 10
    cateogry_min_df = 10
    
    dataset.loc[:, 'category_name'].fillna(value='missing', inplace=True)
    dataset.loc[:, 'brand_name'].fillna(value='missing', inplace=True)
    dataset.loc[:, 'item_description'].fillna(value='missing', inplace=True)
    

    pop_brand = dataset.loc[:, 'brand_name'].value_counts().loc[
            lambda x: x.index != 'missing'].index[:NUM_BRANDS]
    
    dataset.loc[~dataset.loc[:, 'brand_name'].isin(pop_brand),
                'brand_name'] = 'missing'
    

    print('[{}] Finished to handle missing'.format(time.time() - start_time))
    
    dataset.loc[:, 'category_name'] = dataset.loc[:, 'category_name'].astype(
                                                        'category')
    dataset.loc[:, 'item_condition_id'] = dataset.loc[:, 'item_condition_id'
                                               ].astype('category')
    
    print('[{}] Finished to convert categorical'.
          format(time.time() - start_time))
    
    name = dataset['name']
    category_name = dataset['category_name']
    item_description = dataset['item_description']
    brand_name = dataset['brand_name']
    item_shipping = dataset[['item_condition_id', 'shipping']]
    del dataset
    gc.collect()
    
    cv = CountVectorizer(min_df=NAME_MIN_DF)
    X_name = cv.fit_transform(name)
    print('[{}] Finished count vectorize `name`'.
          format(time.time() - start_time))
    
    del name
    gc.collect()
    
    cv = CountVectorizer(min_df=cateogry_min_df)
    X_category = cv.fit_transform(category_name)
    print('[{}] Finished count vectorize `category_name`'.
          format(time.time() - start_time))
    
    del category_name
    del cv
    gc.collect()

    '''
    NOTES
    
    # Why not use TfidfVectorizer for 'name' and 'category_name' feature as well.
    # The reason is that the frequency of each word does not matter in 
    # name and category_name.  
    # Only creating dummy variables for each name and category_name.  
    # 
    # n-gram is a contiguous sequence of n items from a given sample of text 
    # or speech.
    '''
    tv = TfidfVectorizer(max_features=None,
                         ngram_range=(1, 1),
                         stop_words='english')
    X_description = tv.fit_transform(item_description)
    
    del item_description
    del tv
    gc.collect()
    
    print('[{}] Finished TFIDF vectorize `item_description`'.
          format(time.time() - start_time))
    
    # one-hot encoding.  
    lb = LabelBinarizer(sparse_output=True)
    X_brand = lb.fit_transform(brand_name)
    
    del brand_name
    del lb
    gc.collect()
    
    print('[{}] Finished label binarize `brand_name`'.
          format(time.time() - start_time))
    
    # The benefit of compressesd sparse row matrix is to reduce time and 
    # space complexity.  
    X_dummies = csr_matrix(pd.get_dummies(item_shipping,
                                          sparse=True).values)
    
    del item_shipping
    gc.collect()
    
    print('[{}] Finished to get dummies on `item_condition_id` and `shipping`'.
          format(time.time() - start_time))
    
    # Stacking matrices
    sparse_merge = hstack((X_dummies, X_description, X_brand, X_category, X_name
                           )).tocsr()
    print('[{}] Finished to create sparse merge'.
          format(time.time() - start_time))

    return sparse_merge


def root_mean_squared_logarithmic_error(y_true, y_pred):
    first_log = log(y_pred + 1)
    second_log = log(y_true + 1)
    return sqrt(mean(square(first_log - second_log)))

def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean(square(y_pred - y_true)))

def boosting_model(prepped_train_x, train_y, prepped_test_x, test_y):
    params = {
        'learning_rate':.01,
        'objective':'regression',
        'num_leaves':20,
        'metric': 'RMSE',
        'feature_fraction':.3}
    
    d_train = lgb.Dataset(prepped_train_x, label=train_y)
    model = lgb.train(params, train_set=d_train, num_boost_round=4000)
    
    pred_train = model.predict(prepped_train_x)
    pred_test = model.predict(prepped_test_x)
    
    # Training RMSE
    print('Training RMSE', root_mean_squared_logarithmic_error(train_y, pred_train))
    
    # Test RMSE
    print('Validation RMSE', root_mean_squared_logarithmic_error(test_y, pred_test))
    return model

def lasso_model(prepped_train_x, train_y, prepped_test_x, test_y):
    model = Lasso(fit_intercept=True, normalize=True, alpha=0, max_iter=200)
    model.fit(prepped_train_x, train_y)
    pred_train = model.predict(prepped_train_x)
    # Training RMSE
    print('Training RMSE', root_mean_squared_logarithmic_error(train_y, pred_train))
    pred_test = model.predict(prepped_test_x)
    
    # Test RMSE
    print('Validation RMSE', root_mean_squared_logarithmic_error(test_y, pred_test))
    
    return model

def ridge_model(prepped_train_x, train_y, prepped_test_x, test_y):
    model = Ridge(alpha=700, solver="sag", fit_intercept=True, random_state=205)
    model.fit(prepped_train_x, train_y)
    pred_train = model.predict(prepped_train_x)
    # Training RMSE
    print('Training RMSE', root_mean_squared_error(train_y, pred_train))
    pred_test = model.predict(prepped_test_x)
    
    # Test RMSE
    print('Validation RMSE', root_mean_squared_error(test_y, pred_test))
    
    return model

def lasso_tuning(prepped_train_x, train_y):
    # Takes a day to run.  
    model = Lasso(fit_intercept=True, normalize=True, max_iter=200)
    alphas= array([1, .9, .8, .7, .6, .5, .4, .3, .2, .1, .001, 0])
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(prepped_train_x, train_y)
    print(grid.best_params_)

def ridge_tuning(prepped_train_x, train_y):
    model = Ridge(fit_intercept=True, normalize=True, max_iter=200)
    alphas= array([i for i in range(500, 1000, 1)])
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    grid.fit(prepped_train_x, train_y)
    print(grid.best_params_)

def main():
    data_1 = pd.read_table('../input/train.tsv',
                          quoting=csv.QUOTE_NONE, encoding='utf-8')
    data_1 = data_1.sample(frac=1).reset_index(drop=True)
#    data_2 = pd.read_table('../input/test.tsv', quoting=csv.QUOTE_NONE,
#                           encoding='utf-8')
    data_3 = pd.read_table('../input/test_stg2.tsv', quoting=csv.QUOTE_NONE,
                           encoding='utf-8')
    # Doing RMSLE
    y = log1p(data_1["price"])
    train_num_row = data_1.shape[0]
    merged_data = pd.concat([data_1, data_3], sort=True)
    test_id = data_3.loc[:, 'test_id']
    
    del data_1
    del data_3
    gc.collect()
    
    train_cutoff = int(train_num_row * .7)
    prepped_data = preprocessing(merged_data)
    
    del merged_data
    gc.collect()
    
    prepped_train = prepped_data[:train_num_row]
    prepped_test = prepped_data[train_num_row:]
    train_x = prepped_train[:train_cutoff]
    valid_x = prepped_train[train_cutoff:]
    
    del prepped_train
    gc.collect()
    
    train_y = y[:train_cutoff]
    valid_y = y[train_cutoff:]
    
    optimized_ridge = ridge_model(train_x, train_y,  valid_x, valid_y)
    optimized_gbm = boosting_model(train_x, train_y, valid_x, valid_y)w
    
    pred = .4*optimized_ridge.predict(prepped_test)
    pred += .6*optimized_gbm.predict(prepped_test)
    
    submission = pd.DataFrame({'test_id' : test_id,
                              'price' : expm1(pred)})
    
    submission.to_csv("submission.csv", index=False)
    
if __name__ == '__main__':
    main()