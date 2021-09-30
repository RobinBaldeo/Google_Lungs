# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# this is a test
import pandas as pd


def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    for i in [train, test]:
        print()
        print(i.dtypes)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
