import pandas as pd
import numpy as np
from optuna.integration import LightGBMPruningCallback
import sklearn.model_selection as ms
from sklearn.linear_model import LogisticRegression
pd.set_option('expand_frame_repr', False)
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
import optuna
from collections import namedtuple
import  sqlite3


import json
from sklearn.linear_model import SGDClassifier

class model_best_para:
    def __init__(self, xtrain, xval, ytrain, yval, num_iter):
        self.xtrain = xtrain
        self.xval = xval
        self.yval = yval
        self.ytrain = ytrain
        self.num_iter = num_iter

    def rf_gosss_gbdt(self, trial, x, y):
        xtrain, xval = x
        ytrain, yval, j = y

        fixed_para = {
            "early_stopping_rounds": 5,
            'eval_metric': 'auc',
            'eval_set': [(xval, yval)],
            'verbose': 25,
            'callbacks': [LightGBMPruningCallback(trial, 'auc')],
        }

        param = {
            "objective": trial.suggest_categorical("objective", ["binary","cross_entropy"]),
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": j,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500, log=True),
            # "max_depth":trial.suggest_int("max_depth", 1, 8, log=True),
            # "learning_rate": trial.suggest_float("learning_rate", 0.01, 1., log=True),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 5., log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 5., log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }

        if param["boosting_type"] in ['rf', 'gbdt']:
            param["feature_fraction"] = trial.suggest_float("feature_fraction", 0.4, 1.0)
            param["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.4, 1.0)
            param["bagging_freq"] = trial.suggest_int("bagging_freq", 1, 7)

        gbm = LGBMClassifier(**param)
        gbm.fit(xtrain, ytrain, **fixed_para)
        preds = gbm.predict_proba(xval)[:, 1]

        return roc_auc_score(yval, preds)


    def para(self):
        para = namedtuple("para", "model_ best_para m_score")
        lst = []
        for n in ['goss', 'gbdt', 'rf']:
            study = optuna.create_study(direction='maximize', study_name="robin")
            study.optimize(lambda i: self.rf_gosss_gbdt(i, (self.xtrain, self.xval), (self.ytrain, self.yval, n)),
                           n_trials=self.num_iter)

            lst.append(para(model_=n, best_para=json.dumps(study.best_trial.params),
                            m_score=([j for i, j in study.best_trial.intermediate_values.items()])[-1]))
            # print(f"best trial {n}: {([j for i, j in study.best_trial.intermediate_values.items()])[-1]}")
            # print(f"Trial #{study.best_trial.number} best trial: {study.best_trial.params}")

        # adding parameters to sql lite
        with sqlite3.connect("db.sep_tub.DB") as conn:
            tbl = pd.DataFrame(lst)
            tbl.to_sql("base_para", conn, if_exists="replace")


        return lst


class build_base(model_best_para):
    def __init__(self, X_, Y_,test, num_iter, fold):
        self.x, self.weight_x, self.y, self.weight_y = ms.train_test_split(X_, Y_, test_size=.05, shuffle=True, random_state=0)

        self.test = test
        self.xval = pd.DataFrame(self.test.drop(columns=["id"]))

        self.folds = fold
        self.df_split = ms.StratifiedKFold(n_splits=self.folds, shuffle=True)
        # super().__init__(self.x, self.weight_x, self.y, self.weight_y, num_iter)

    # def dummy(self):
    #     super().para()
    #     pass