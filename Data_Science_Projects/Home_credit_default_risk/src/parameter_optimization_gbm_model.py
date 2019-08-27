import lightgbm as lgb
import random
import numpy as np
import pandas as pd
import preprocessing
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt.pyll.stochastic import sample
from hyperopt import fmin
import warnings

warnings.filterwarnings("ignore")

MAX_EVALS = 2
data = pd.read_csv("../input/application_train_updated.csv")
y = data.loc[:, "TARGET"]
data.drop("TARGET", axis=1, inplace=True)
data = preprocessing.categorical_to_dummy(data)
final_x, final_y = preprocessing.smotenc_oversampling(data, y)
params_hyperopt = {
'metric': 'auc',
'objective': 'binary',
'num_leaves': hp.quniform('num_leaves', 5, 200, 1),
'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
"feature_fraction": hp.uniform('feature_fraction', 0, 1),
#'lambda_l1': hp.uniform('lambda_l1', .001, 100)
}
param_grid = {
"feature_fraction": list(np.linspace(0, 1)),
'num_leaves': list(range(20, 150)),
'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
}


def boosting_cv(hyperparameters):
    number_of_folds = 10
    metric_mean = hyperparameters["metric"] + "-mean"
    train_data = lgb.Dataset(data, label=y)
    results = lgb.cv(
        hyperparameters, 
        train_data, 
        nfold=number_of_folds, 
        early_stopping_rounds=100,
        num_boost_round=2000
        
    )
    cv_score = results[metric_mean][-1]
    loss = 1 - cv_score
    best_num_boosting_tree = len(results[metric_mean])
    hyperparameters['n_estimators'] = best_num_boosting_tree 
    return {'auc_cv_score':cv_score,
            'loss':loss,
            'parameters':hyperparameters,
            'status':STATUS_OK}


def random_search(param_grid):
    number_of_rounds = 2000
    results = pd.DataFrame(columns = ['auc_cv_score', 'parameters', 'iteration'],
                                  index = list(range(MAX_EVALS)))
    
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters["objective"] = "binary"
        hyperparameters["metric"] = "auc"
        hyperparameters["num_boost_round"] = number_of_rounds
        # Evaluate randomly selected hyperparameters
        eval_results = boosting_cv(hyperparameters)
        eval_results['iteration'] = i
        del eval_results['loss']
        del eval_results['status']
        results.loc[i, :] = eval_results
    
    # Sort with best score on top
    results.sort_values('auc_cv_score', ascending = False, inplace = True)
    results.reset_index(inplace = True)
    return results 


def parameter_optimzation(space):
    # Sample from the full space
    x = sample(space)
    x['num_leaves'] = int(x['num_leaves'])
    # Create the parameter optimization algorithm.  
    tpe_algorithm = tpe.suggest
    # Record results
    trials = Trials()
    # Run optimization
    fmin(fn = boosting_cv, space = x, algo = tpe_algorithm,
         trials = trials, max_evals = MAX_EVALS)
    # Getting highest auc
    trials = sorted(trials.results, key = lambda x: x['loss'])
    return trials[:1]


def main():
    rounds = 10
    res = []
    curr_auc = 0
    index = 0
    for i in range(rounds):
        best = parameter_optimzation(params_hyperopt)
        res.append(best[0])
    for i, ele in enumerate(res):
        if ele['auc_cv_score'] > curr_auc:
            index = i
    print("Best parameters:", res[index])
    
#    random_results = random_search(param_grid)
#    print('The best auc cv score is {:.5f}'.format(random_results.loc[0, 'auc_cv_score']))
#    print('\nThe best hyperparameters are:\n')
#    print(random_results.loc[0, 'parameters'])
    

if __name__ == "__main__":
    main()
