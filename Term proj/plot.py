import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
import numpy as np


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


def violin(df, x, y, t):
    ax = sns.violinplot(x=x, y=y, data=df).set_title(t)
    plt.show()
    plt.savefig(t+'.png')


def convert(df):
    df['zero_crossing_rate'] = df['zero_crossing_rate'].apply(lambda x: [float(e) for e in x.split('/')])
    df['spectral_centroid'] = df['spectral_centroid'].apply(lambda x: [float(e) for e in x.split('/')])
    df['spectral_rolloff'] = df['spectral_rolloff'].apply(lambda x: [float(e) for e in x.split('/')])
    return df


def general(df, x, t):
    if x == 'length':
        violin(df[(df.artist != 'magenta')], "by", x, t)
        violin(df[(df.artist != 'magenta') & (df.by == 'AI')], "artist", x, "AI." + t)
        violin(df[(df.artist != 'magenta') & (df.by == 'human')], "artist", x, "Human." + t)
    else:
        violin(df, "by", x, t)
        violin(df[df.by == 'AI'], "artist", x, "AI." + t)
        violin(df[df.by == 'human'], "artist", x, "Human." + t)


def stack(df,x,g,t):
    x_var = x
    groupby_var = g
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]
    plt.figure(figsize=(16, 9), dpi=80)
    colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False,
                                color=colors[:len(vals)])
    plt.legend({group: col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')
    plt.show()
    plt.savefig(t+'.png')

if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    data = convert(data)
    violin(data, "artist", "length", "Length(ms)")
    ax = sns.violinplot(x=data["tempo"]).set_title('tempo')
    plt.show()
