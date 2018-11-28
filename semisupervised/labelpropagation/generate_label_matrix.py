from pyspark.sql import functions as F
from pyspark.mllib.linalg import distributed
from functools import partial


def generate_label_matrix(df, label_col='label',
                          id_col='id', k=None):
    null_nan_check = (
            ~F.isnan(F.col(label_col)) &
            ~F.isnull(F.col(label_col))
    )
    y_known = (df.filter(null_nan_check).select(id_col, label_col).cache())
    # y_known.select(label_col).distinct().show()

    y_unknown = (df.filter(~null_nan_check).select(id_col).cache())
    if k:
        y_range = (y_known.select(label_col).distinct().count())
    else:
        y_range = int(k)

    y_known_rdd = y_known.rdd.map(
        lambda x: distributed.MatrixEntry(
            i=int(x[id_col]),
            j=int(x[label_col]),
            value=1.0
        )
    )
    y_unknown_rdd = y_unknown.rdd.flatMap(
        lambda x: [distributed.MatrixEntry(
            i=int(x[id_col]),
            j=idx,
            value=.50
        ) for idx in range(int(y_range))]
    )
    y_matrix = distributed.CoordinateMatrix(
        entries=y_unknown_rdd.union(y_known_rdd),
        numRows=df.count(),
        numCols=y_range
    )
    y_unknown.unpersist()
    y_known.unpersist()
    return y_known_rdd, y_matrix


def create_label_matrix(df, **kwargs):
    # Bekskrivelse: Denne metode danner label matrix for data.
    # input: df
    # output: y_known_rdd, y_matrix

    id_col = kwargs.get('id_col', None)
    label_col = kwargs.get('label_col', 'label')
    k = kwargs.get('k', None)
    part_generate_label_matrix = partial(generate_label_matrix, df=df, label_col=label_col, k=k)
    if id_col:  # same as usual
        clamped_y_rdd, initial_y_matrix = part_generate_label_matrix(id_col=id_col)
    else:
        clamped_y_rdd, initial_y_matrix = part_generate_label_matrix(id_col='label_a')
    return clamped_y_rdd, initial_y_matrix
