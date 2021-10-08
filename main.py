# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# this is a test
import pandas as pd
import numpy as np
import  sklearn.model_selection  as ms
from tuningPara import model_best_para




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


    unique_breath_id = [i for i in train['breath_id'].drop_duplicates()]
    split_X = pd.DataFrame(unique_breath_id, index=unique_breath_id, columns=["breath_id"])
    # print(len(split_X.index))

    kf = ms.KFold(n_splits=fold, random_state=None, shuffle=False)

    for counter, (train_index, test_index) in enumerate(kf.split(split_X)):
        hv = len(train_index)
        if counter < fold - 1:
            hv = len(train_index) / fold
            hv = int(hv * (counter + 1))

        xtrain = train.merge(split_X[0:hv], right_on=["breath_id"], left_on=["breath_id"])
        ytrain = xtrain["pressure"]
        xtrain.drop(columns = ["pressure", "breath_id", "id"], inplace=True)
        print(ytrain.shape)
        print(xtrain.shape)

        xval = train.merge(split_X.iloc[test_index, :], right_on=["breath_id"], left_on=["breath_id"])
        yval = xval["pressure"]
        xval.drop(columns = ["pressure", "breath_id", "id"], inplace=True)

        # , X_, Y_, xval, yval
        # , xtrain, xval, ytrain, yval
        tp = model_best_para(xtrain.values, xval.values,ytrain.values, yval.values )
        tp.para()












if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
