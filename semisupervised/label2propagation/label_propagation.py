from pyspark.sql import DataFrame
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import RowMatrix
from shared.WorkflowLogger import logger_info_decorator
import numpy as np
from split_to_submatrix import to_submatries


@logger_info_decorator
def label_propagation(sc: SparkContext, data_frame: DataFrame, *args, **kwargs):
    # Bekskrivelse: Denne metode skal bruges til at samle de dele fra den direkte metode til label propagation
    # input: data_frame - data_frame[id: int, features: vector, label: int]
    # id - data punkt identifikation
    # features - 1 x n vector/matrix med afstande mellem i og j.
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
        # TODO Use method to compute T
        None

    known_lab = data_frame.count() - data_frame.filter(F.isnan(label) or F.isnull(label)).count()

    broad_l = sc.broadcast(known_lab)
    T_ul, T_uu = to_submatries(df=data_frame, broadcast_l=broad_l, **kwargs)
    # TODO Compute: I-Tuu

    # TODO Compute (I-Tuu)^-1

    # TODO Compute (I-Tuu)^-1 + TulYL = Yu

    # TODO Append {YL, YU} to output as corrected label

    identity = np.eye(data_frame.count())
    identity_rdd = sc.parallelize(identity.tolist())
    Identity_mat = RowMatrix(identity_rdd)




    output = None
    
    return output