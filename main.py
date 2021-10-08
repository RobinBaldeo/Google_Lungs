# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# this is a test
import pandas as pd
import numpy as np
import  sklearn.model_selection  as ms
from sklearn.metrics import mean_squared_error, r2_score
from tuningPara import model_best_para
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from collections import namedtuple




def reduce_df(df):

    print(f"orginal dataset :{df.memory_usage().sum() / 1024 ** 2} mb")
    for i in df.columns:
        col_type = df[i].dtypes

        if str(col_type)[0:1] in ["i", "f"]:
            col_min, col_max = np.min(df[i]), np.max(df[i])
            if str(col_type)[0:1] == "i":
                for j in [np.int8,np.int16,np.int32, np.int64]:
                    if col_min > np.iinfo(j).min and col_max < np.iinfo(j).max:
                        df[i] = df[i].astype(j)
                        break
            else:
                for j in [np.float16,np.float32,np.float64]:
                    if col_min > np.finfo(j).min and col_max < np.finfo(j).max:
                        df[i] = df[i].astype(j)
                        break

    print(f"dataset reduced to :{df.memory_usage().sum() / 1024 ** 2} mb")
    return df



def main():
    train = reduce_df(pd.read_csv('train.csv'))
    test = reduce_df(pd.read_csv('test.csv'))
    fold = 10
    pred_ = namedtuple("pred_", "pred actual")
    scale  = StandardScaler()
    # train_ = train.copy()
    train["row"] = (train.groupby('breath_id')['id'].rank(method="first", ascending=True)).astype("int")
    test["row"] = (test.groupby('breath_id')['id'].rank(method="first", ascending=True)).astype("int")
    X = train.drop(columns=["pressure", 'id', 'breath_id'])
    Y = pd.DataFrame(train["pressure"])
    X = scale.fit_transform(X)

    xtrain, xtest,ytrain, ytest = ms.train_test_split(X, Y, test_size=.2, random_state=0, shuffle=True)

    print(xtrain)
    print(xtest)
    #
    model = KNeighborsRegressor(n_neighbors=9)
    model.fit(xtrain, ytrain.values)
    pred = model.predict(xtest)

    df = pd.DataFrame(ytest.values)
    df["p"] = pred

    print(df.head(5))
    df.columns = ["actual", "predicted"]
    print(df.head(20))





    print(mean_squared_error(ytest.values, pred )**.5)




















if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
