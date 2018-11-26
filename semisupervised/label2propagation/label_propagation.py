from pyspark.sql import DataFrame
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from shared.WorkflowLogger import logger_info_decorator
import numpy as np
from split_to_submatrix import to_submatries
from equation import compute_equation
from compute_distances import compute_distances


@logger_info_decorator
def label_propagation(sc: SparkContext, data_frame: DataFrame, *args, **kwargs):
    # Bekskrivelse: Denne metode skal bruges til at samle de dele fra den direkte metode til label propagation
    # input: data_frame - data_frame[id: int, features: vector, label: int]
    # id - data punkt identifikation
    # features - 1 x n vector/matrix med afstande mellem i og j. Eller punkter i n-dim rum
    # label - label med kendte og ukendte labels
    # output: output - data_frame[id: int, features: vector, label: int, corrected_label: int]

    # definere features, label og id
    # TODO Do refactor to method
    features = kwargs.get('features', None)
    assert features in data_frame.columns, "Error {feature} is not in dataframe".format(feature=features)

    id = kwargs.get("id", "id")
    assert id in data_frame.columns, "Error {id} is not amoung dataframe columns {cols}".format(
        id=id,
        cols=", ".join(data_frame.columns)
    )

    label = kwargs.get("label", "label")
    assert label in data_frame.columns, "Error {label} is not amoung dataframe columns {cols}".format(
        label=label,
        cols=", ".join(data_frame.columns)
    )

    sqlCtx = SQLContext(sc)

    # Define T Matrix as the Probability matrix
    precomputed_T = kwargs.get('precomputed_distance', False)
    if not precomputed_T:
        # Use method to compute T
        distances_df = compute_distances(sc=sc, data_frame=data_frame, id=id, label_col=label, feature_col=features)
    else:
        distances_df = data_frame

    unknown_lab = distances_df.filter(F.isnan(label) | F.isnull(label)).count()
    known_lab = distances_df.count() - unknown_lab

    # Split T into for sub-matrices
    broad_l = sc.broadcast(known_lab)
    broad_u = sc.broadcast(unknown_lab)
    T_ll_df, T_lu_df, T_ul_df, T_uu_df = to_submatries(df=distances_df, broadcast_l=broad_l, **kwargs)

    # Do computations
    output = compute_equation(sc=sc, T_uu=T_uu_df, T_ll=T_ll_df, T_ul=T_ul_df,u_broadcast=broad_u, **kwargs)
    return output