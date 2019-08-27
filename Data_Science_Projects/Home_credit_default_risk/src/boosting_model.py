import lightgbm as lgb
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import preprocessing
import warnings

warnings.filterwarnings("ignore")


def boosting_model(best_params, prepped_train_x, train_y):
    d_train = lgb.Dataset(prepped_train_x, label=train_y)
    model = lgb.train(best_params, train_set=d_train)
    pred_train = model.predict(prepped_train_x)
    train_fpr, train_tpr, train_thresholds = roc_curve(train_y, pred_train, pos_label=1)
    print("Training auc", auc(train_fpr, train_tpr))
    return model


def feature_importance(model):
    FI_measure = model.feature_importance(importance_type="gain")
    features = model.feature_name()
    temp = pd.DataFrame(
        sorted(zip(FI_measure, features), reverse=True),
        columns=["Feature Importance", "Feature Name"],
    )
    temp = temp.loc[0:50, :]

    plt.figure(figsize=(20, 10))
    sns.barplot(
        x="Feature Importance",
        y="Feature Name",
        data=temp.sort_values(by="Feature Importance", ascending=False),
    )
    plt.title("Top 50 LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.show()


def main():
    data = pd.read_csv("../input/application_train_updated.csv")
    learn_rate = .5
    number_of_leaves = 200
    fraction_of_features = 0.3
    number_of_rounds = 700
    params = {
        "learning_rate": learn_rate,
        "objective": "binary",
        "num_leaves": number_of_leaves,
        "metric": "auc",
        "feature_fraction": fraction_of_features,
        "num_boost_round": number_of_rounds,
    }
    y = data.loc[:, "TARGET"]
    data.drop("TARGET", axis=1, inplace=True)
    data = preprocessing.categorical_to_dummy(data)
    final_x, final_y = preprocessing.smotenc_oversampling(data, y)
    gbm = boosting_model(params, final_x, final_y)
    feature_importance(gbm)
    gbm.save_model("../output/gradient_boosting_model.txt")


if __name__ == "__main__":
    main()
