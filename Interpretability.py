
import pandas as pd 
import shap

category = 'Candy & Chocolate'

# load best model
model = pd.read_pickle('models/best_performing_model_Candy & Chocolate_2022-04-29.sav') # catboost-regressor

# load data 
df_train = pd.read_csv('data/' + category + '/df_train_lda.csv')
df_test = pd.read_csv('data/' + category + '/df_test_lda.csv')
X_train = df_train.drop(columns='avg_rating')
y_train = df_train['avg_rating']
X_test = df_test.drop(columns='avg_rating')
y_test = df_test['avg_rating']

# load df_train and df_test
categorical_features_indices = [] 
shap_values = model.get_feature_importance(Pool(X_test, label=y_test,cat_features=), type='ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]
shap.initjs()
shap.force_plot(expected_value, shap_values[3,:], X_test.iloc[3,:])