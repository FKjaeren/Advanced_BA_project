import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
# 8 most popular regression models
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, SGDRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import pickle
from datetime import date


# set random seed
np.random.seed(42)

# load data
category = 'beverages'
if category == 'all':
        # load data 
        df = pd.read_csv('data/lda_and_preprocessed_df.csv')
        # get dummies and drop description
        df = pd.get_dummies(df, columns=['category','top_brand'])
        df = df.drop(columns=['description','std_rating'])
else:
        # load data 
        df = pd.read_csv('data/df_'+category+'_with_lda.csv')
        # get dummies and drop description
        df = pd.get_dummies(df, columns=['top_brand'])
        df = df.drop(columns=['category','description','std_rating'])
df = df.dropna()

# prepare for training
y = df['avg_rating']
X = df.drop(columns=['avg_rating'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

def train_regression_models(X_train, X_test, y_train, y_test):
    # Linear Regression
    linear_regression = LinearRegression().fit(X_train, y_train)
    y_linear_regression = linear_regression.predict(X_test)
    MAE_linear_regression = mean_absolute_error(y_test, y_linear_regression)
    r2_linear_regression = r2_score(y_test, y_linear_regression)
    var_linear_regression = explained_variance_score(y_test, y_linear_regression)
    print("----------------------")
    print("Linear Regression: ")
    print("MAE ", MAE_linear_regression)
    print("R2 ", r2_linear_regression)
    print("Explained variance ", var_linear_regression)
    print("----------------------")

    # XGBoost Regressor
    xgb_regressor = XGBRegressor().fit(X_train, y_train)
    y_xgb_regressor = xgb_regressor.predict(X_test)
    MAE_xgb_regressor = mean_absolute_error(y_test, y_xgb_regressor)
    r2_xgb_regressor = r2_score(y_test, y_xgb_regressor)
    var_xgb_regressor = explained_variance_score(y_test, y_xgb_regressor)
    print("----------------------")
    print("XGBoost Regressor: ")
    print("MAE ", MAE_xgb_regressor)
    print("R2 ", r2_xgb_regressor)
    print("Explained variance ", var_xgb_regressor)
    print("----------------------")

    # CatBoost Regressor
    catboost_regressor = CatBoostRegressor(allow_writing_files=False).fit(X_train, y_train, logging_level='Silent')
    y_catboost_regressor = catboost_regressor.predict(X_test)
    MAE_catboost_regressor = mean_absolute_error(y_test, y_catboost_regressor)
    r2_catboost_regressor = r2_score(y_test, y_catboost_regressor)
    var_catboost_regressor = explained_variance_score(y_test, y_catboost_regressor)
    print("----------------------")
    print("CatBoost Regressor: ")
    print("MAE ", MAE_catboost_regressor)
    print("R2 ", r2_catboost_regressor)
    print("Explained variance ", var_catboost_regressor)
    print("----------------------")

    # Stochastic Gradient Descent Regression
    sgd_regressor = SGDRegressor().fit(X_train, y_train)
    y_sgd_regressor = sgd_regressor.predict(X_test)
    MAE_sgd_regressor = mean_absolute_error(y_test, y_sgd_regressor)
    r2_sgd_regressor = r2_score(y_test, y_sgd_regressor)
    var_sgd_regressor = explained_variance_score(y_test, y_sgd_regressor)
    print("----------------------")
    print("Stochastic Gradient Descent Regression: ")
    print("MAE ", MAE_sgd_regressor)
    print("R2 ", r2_sgd_regressor)
    print("Explained variance ", var_sgd_regressor)
    print("----------------------")

    # # Kernel Ridge Regression
    # kernel_rigde = KernelRidge().fit(X_train, y_train)
    # y_kernel_rigde = kernel_rigde.predict(X_test)
    # MAE_kernel_rigde = mean_absolute_error(y_test, y_kernel_rigde)
    # r2_kernel_rigde = r2_score(y_test, y_kernel_rigde)
    # var_kernel_rigde = explained_variance_score(y_test, y_kernel_rigde)
    # print("----------------------")
    # print("Kernel Ridge Regression: ")
    # print("MAE ", MAE_kernel_rigde)
    # print("R2 ", r2_kernel_rigde)
    # print("Explained variance ", var_kernel_rigde)
    # print("----------------------")

    # Elastic Net Regression
    elastic_net = ElasticNet().fit(X_train, y_train)
    y_elastic_net = elastic_net.predict(X_test)
    MAE_elastic_net = mean_absolute_error(y_test, y_elastic_net)
    r2_elastic_net = r2_score(y_test, y_elastic_net)
    var_elastic_net = explained_variance_score(y_test, y_elastic_net)
    print("----------------------")
    print("Elastic Net Regression: ")
    print("MAE ", MAE_elastic_net)
    print("R2 ", r2_elastic_net)
    print("Explained variance ", var_elastic_net)
    print("----------------------")

    # Bayesian Ridge Regression
    bayesian_ridge = BayesianRidge().fit(X_train, y_train)
    y_bayesian_ridge = bayesian_ridge.predict(X_test)
    MAE_bayesian_ridge = mean_absolute_error(y_test, y_bayesian_ridge)
    r2_bayesian_ridge = r2_score(y_test, y_bayesian_ridge)
    var_bayesian_ridge = explained_variance_score(y_test, y_bayesian_ridge)
    print("----------------------")
    print("Bayesian Ridge Regression: ")
    print("MAE ", MAE_bayesian_ridge)
    print("R2 ", r2_bayesian_ridge)
    print("Explained variance ", var_bayesian_ridge)
    print("----------------------")

    # Gradient Boosting Regression
    gb_regressor = GradientBoostingRegressor().fit(X_train, y_train)
    y_gb_regressor = gb_regressor.predict(X_test)
    MAE_gb_regressor = mean_absolute_error(y_test, y_gb_regressor)
    r2_gb_regressor = r2_score(y_test, y_gb_regressor)
    var_gb_regressor = explained_variance_score(y_test, y_gb_regressor)
    print("----------------------")
    print("Gradient Boosting Regression: ")
    print("MAE ", MAE_gb_regressor)
    print("R2 ", r2_gb_regressor)
    print("Explained variance ", var_gb_regressor)
    print("----------------------")

    MAEs = [MAE_linear_regression, MAE_xgb_regressor, MAE_catboost_regressor, MAE_sgd_regressor,
            MAE_elastic_net, MAE_bayesian_ridge, MAE_gb_regressor]
    models = [linear_regression, xgb_regressor, catboost_regressor, sgd_regressor, elastic_net, 
            bayesian_ridge, gb_regressor]
    best_idx = np.argmin(MAEs)

    return models[best_idx]

# train models
model = train_regression_models(X_train, X_test, y_train, y_test)

predictions = model.predict(X_test)

print("The predictions using the best performing models are: ", predictions)
print("The true values are: ", y_test)
print("This gives the difference between predictions and true values: ", predictions-y_test)
print("The MAE of the model is: ", mean_absolute_error(y_test, predictions))

today = date.today()

if category == 'all':
        filename = 'models/best_performing_model_'+str(today)+'.sav'
        pickle.dump(model, open(filename, 'wb'))
else:
        filename = 'models/best_performing_model_'+category+'_'+str(today)+'.sav'
        pickle.dump(model, open(filename, 'wb'))
