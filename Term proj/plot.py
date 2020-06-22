import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby


def stat(x):
    return pd.Series([
        x.count(), x.mode(), x.min(), x.max(),
        x.quantile(.25), x.median(), x.quantile(.75),
        round(x.mean(), 2), x.var(), round(x.std(), 2), round(x.skew(), 3), round(x.kurt(), 3),
        x.quantile(.75) - x.quantile(.25)
    ], index=[
        '總計', '眾數', '最小值', '最大值',
        '25%分位數', '中位數', '75%分位數',
        '均值', '方差', '標準差', '偏度', '峰度', 'IQR'
    ])


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    ax = sns.violinplot(x="album",y="length", data=data).set_title("Length(ms)")
    plt.show()
