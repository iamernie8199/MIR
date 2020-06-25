import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import groupby
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pywaffle import Waffle


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

def count(df):
    df_c = df.groupby('artist').size().reset_index(name='counts')
    n_categories = df_c.shape[0]
    colors = [plt.cm.tab20b(i / float(n_categories)) for i in range(n_categories)]
    fig = plt.figure(
        FigureClass=Waffle,
        plots={
            '111': {
                'values': df_c['counts'],
                'labels': ["{0} ({1})".format(n[0], n[1]) for n in df_c[['artist', 'counts']].itertuples()],
                'legend': {'loc': 'best', 'bbox_to_anchor': (1, 1), 'fontsize': 18},
                'title': {'label': 'Data', 'loc': 'center', 'fontsize': 30}
            },
        },
        rows=df_c.shape[0],
        colors=colors,
        figsize=(27, 6)
    )
    plt.savefig('data_count.png')
    plt.show()


def violin(df, x, y, t):
    ax = sns.violinplot(x=x, y=y, data=df).set_title(t)
    plt.savefig(t+'.png')
    plt.show()


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
    plt.savefig(t+'.png')
    plt.show()


def lower_diag_matrix_plot(matrix, title=None):
    """ Args:
        matrix - the full size symmetric matrix of any type that is lower diagonalized
        title - title of the plot
    """
    plt.style.use('default')

    # Create lower triangular matrix to mask the input matrix
    triu = np.tri(len(matrix), k=0, dtype=bool) == False
    matrix = matrix.mask(triu)
    fig, ax = plt.subplots(figsize=(20, 20))
    if title:
        fig.suptitle(title, fontsize=32, verticalalignment='bottom')
        fig.tight_layout()
    plot = ax.matshow(matrix)

    # Add grid lines to separate the points
    # Adjust the ticks to create visually appealing grid/labels
    # Puts minor ticks every half step and bases the grid off this
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    # Puts major ticks every full step and bases the labels off this
    ax.set_xticks(np.arange(0, len(matrix.columns), 1))
    ax.set_yticks(np.arange(0, len(matrix.columns), 1))
    plt.yticks(range(len(matrix.columns)), matrix.columns)
    # Must put this here for x axis grid to show
    plt.xticks(range(len(matrix.columns)))
    ax.tick_params(axis='both', which='major', labelsize=24)
    # Whitens (transparent) x labels
    ax.tick_params(axis='x', colors=(0, 0, 0, 0))

    # Add a colorbar for reference
    cax = make_axes_locatable(ax)
    cax = cax.append_axes("right", size="5%", pad=0.05)
    cax.tick_params(axis='both', which='major', labelsize=24)
    fig.colorbar(plot, cax=cax, cmap='hot')

    # Get rid of borders of plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.savefig(title+'.png')
    plt.show()
