#loading packages
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle

#Reading train
data = pd.read_csv(r'../input/insurance-premium-prediction/insurance.csv')

# Label Encoding

le = LabelEncoder()
var = ['sex','smoker','region']
for i in var:
    data[i] = le.fit_transform(data[i])

#define target and ID column
X = data.drop('expenses',axis=1)
y = data['expenses']

# parmas recovered from optuna
lgb_params = {'learning_rate': 0.4616962246009375,
              'lambda_l1': 1.0056418409225514e-08,
              'lambda_l2': 0.0034886051949242197,
              'num_leaves': 56,
              'feature_fraction': 0.9947122992121118,
              'bagging_fraction': 0.8304993202512568,
              'bagging_freq': 7,
              'min_child_samples': 38}
    
# Model building
def cross_val(data,target,model,params):
    kf = KFold(n_splits = 10,shuffle = True,random_state = 2021)
    for fold, (train_idx,test_idx) in enumerate(kf.split(data,target)):
        print(f"Fold: {fold}")
        x_train, y_train = data.iloc[train_idx], target.iloc[train_idx]
        x_test, y_test = data.iloc[test_idx], target.iloc[test_idx]

        alg = model(**params,random_state = 2021)
        alg.fit(x_train, y_train,
                eval_set=[(x_test, y_test)],
                early_stopping_rounds=400,
                verbose=False)
        pred = alg.predict(x_test)
        error = mean_squared_error(y_test, pred)
        print(f" mean_squared_error: {error}")
        print("-"*50)
    
    return alg

lgb_model = cross_val(X,y,LGBMRegressor,lgb_params)

# save the model to disk
filename = 'finalized_model.pkl'
pickle.dump(lgb_model, open(filename, 'wb'))
