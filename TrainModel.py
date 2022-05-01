import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score
# 8 most popular regression models
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge, SGDRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle
from datetime import date

# set random seed
np.random.seed(42)

# load data
category = 'Beverages'
if category == 'all':
        # load data 
        df = pd.read_csv('data/lda_and_preprocessed_df.csv')
        # get dummies and drop description
        df = pd.get_dummies(df, columns=['category','top_brand'])
        df = df.drop(columns=['description','std_rating'])
else:
        # load data 
        df_train = pd.read_csv('data/' + category + '/df_train_lda.csv')
        df_test = pd.read_csv('data/' + category + '/df_test_lda.csv')
        # get dummies and drop description

        # CLUSTERING
        # df_train = pd.get_dummies(df_train, columns=['top_brand'])
        df_train = df_train.drop(columns=['description','std_rating'])
        df_test = df_test.drop(columns=['description','std_rating'])
df_train = df_train.dropna()
df_test = df_test.dropna()

# prepare for training
y_train = df_train['avg_rating']
y_test = df_test['avg_rating']
X_train = df_train.drop(columns=['avg_rating'])
X_test = df_test.drop(columns=['avg_rating'])

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
    names = ['linear_regression', 'xgb_regressor', 'catboost_regressor', 'sgd_regressor', 'elastic_net', 
            'bayesian_ridge', 'gb_regressor']
    best_idx = np.argmin(MAEs)

    return models[best_idx], names[best_idx]

def tune_model(model, name, X_train, y_train):
        if name == 'catboost_regressor':
                # tune parameters of catboost
                parameters = {'depth' : [5, 10, 15, 20],
                                'learning_rate' : [0.01, 0.02, 0.03]}
                Grid_CBC = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1, verbose=0)
                Grid_CBC.fit(X_train, y_train)
                depth = Grid_CBC.best_params_['depth']
                learning_rate = Grid_CBC.best_params_['learning_rate']
                catboost_regressor = CatBoostRegressor(allow_writing_files=False, depth=depth, learning_rate=learning_rate).fit(X_train, y_train, logging_level='Silent')
        return Grid_CBC.best_params_, catboost_regressor           


# train and tune model
model, name = train_regression_models(X_train, X_test, y_train, y_test)
# params, tuned_model = tune_model(model, name, X_train, y_train)

# validate model
# predictions = tuned_model.predict(X_test)

# print("The predictions using the best performing models are: ", predictions)
# print("The true values are: ", y_test)
# print("This gives the difference between predictions and true values: ", predictions-y_test)
# print("The MAE of the model is: ", mean_absolute_error(y_test, predictions))

today = date.today()

if category == 'all':
        filename = 'models/best_performing_model_'+str(today)+'.sav'
        pickle.dump(model, open(filename, 'wb'))
else:
        filename = 'models/'+category+'/best_performing_model_'+str(today)+'.sav'
        pickle.dump(model, open(filename, 'wb'))
