import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.svm import SVC
import optuna
from collections import namedtuple
import  sqlite3


import json
from sklearn.linear_model import SGDClassifier

class model_best_para:
    def __init__(self, xtrain, xval, ytrain, yval):
        self.xtrain = xtrain
        self.xval = xval
        self.yval = yval
        self.ytrain = ytrain


    def rf_gosss_gbdt(self, trial, x, y):
        xtrain, xval = x
        ytrain, yval, j = y

        fixed_para = {
            "early_stopping_rounds": 5,
            'eval_metric': 'rmse',
            'eval_set': [(xval, yval)],
            'verbose': 25,
            # 'callbacks': [LightGBMPruningCallback(trial, 'rmse')],
        }

        param = {
        'metric': 'rmse',
        'random_state': 0,
        'n_estimators': 1000,
        # 'device': 'gpu',
        # 'gpu_platform_id': 0,
        # 'gpu_device_id': 0,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.2, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.5),
        'max_depth': trial.suggest_categorical('max_depth', [5,10,20,40,100, -1]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
        }



        gbm = LGBMRegressor(**param)
        gbm.fit(xtrain, ytrain, **fixed_para)
        preds = gbm.predict(xval)

        return mean_squared_error(yval, preds)


    def para(self):
        para = namedtuple("para", "best_para m_score")
        lst = []

        study = optuna.create_study(direction='maximize', study_name="robin")
        study.optimize(lambda i: self.rf_gosss_gbdt(i, (self.xtrain, self.xval), (self.ytrain, self.yval, 5)),
                       n_trials=5)

        # lst.append(para(best_para=json.dumps(study.best_trial.params),
        #                 m_score=([j for i, j in study.best_trial.intermediate_values.items()])[-1]))
        #
        # # adding parameters to sql lite
        # with sqlite3.connect("db.sep_tub.DB") as conn:
        #     tbl = pd.DataFrame(lst)
        #     tbl.to_sql("base_para", conn, if_exists="append")
        #
        #
        # print(lst)


# class build_base(model_best_para):
#     def __init__(self, X_, Y_,xval, yval):
#         super().__init__(X_, Y_, xval,yval)

