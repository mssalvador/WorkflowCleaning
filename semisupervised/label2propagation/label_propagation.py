from pyspark.sql import DataFrame
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import VectorUDT
from pyspark.mllib.linalg import Vectors
from shared.WorkflowLogger import logger_info_decorator
import numpy as np
from split_to_submatrix import to_submatries
from matrix_inversion import invert
from create_label_matrix import label_matrix


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

    unknown_lab = data_frame.filter(F.isnan(label) or F.isnull(label)).count()
    known_lab = data_frame.count() - unknown_lab

    broad_l = sc.broadcast(known_lab)
    broad_u = sc.broadcast(unknown_lab)
    T_ll, T_lu, T_ul, T_uu = to_submatries(df=data_frame, broadcast_l=broad_l, **kwargs)

    # Compute: I-Tuu
    create_sparse_ident = F.udf(lambda x: Vectors.sparse(broad_u.value, [x - broad_l.value], [1.0]), VectorUDT())
    subtract = F.udf(lambda x, y: x - y, VectorUDT())

    identity_df = T_uu.withColumn(colName="Identity", col=create_sparse_ident(F.col(id)))
    I_minus_Tuu = identity_df.select(F.col(id),
                                     subtract(F.col("Identity"), F.col("right")).alias("Iminus"))

    # Compute (I-Tuu)^-1
    inverted_mat = invert(sc=sc, data_frame=I_minus_Tuu, column="Iminus")

    # TODO create Y_l and Y_u as respectfully (l x C) and (u x C) matrices. C is the number of class'

    # TODO Compute (I-Tuu)^-1 + TulYL = Yu
    T_ul
    # TODO Append {YL, YU} to output as corrected label

    identity = np.eye(data_frame.count())
    identity_rdd = sc.parallelize(identity.tolist())
    Identity_mat = RowMatrix(identity_rdd)




    output = None
    
    return output