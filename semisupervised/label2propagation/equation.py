from pyspark.sql import DataFrame
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import Row
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow

from semisupervised.label2propagation.matrix_inversion import invert
from create_label_matrix import label_matrix

def compute_equation(
        sc: SparkContext,
        T_uu: DataFrame=None,
        T_ll: DataFrame=None,
        T_ul: DataFrame=None,
        u_broadcast=None,
        **kwargs):
    # Computes the direct equation from the article (I-T_uu)^-1T_ulY_l
    # Input: sc: SparkContext
    # Input: T_uu: DataFrame[n_id, id, label, feature (distances), right vector] - submatrix Tuu containing distances between unknown labels
    # Input: T_ll: DataFrame[n_id, id, label, feature (distances), left vector] - submatrix Tll Used for getting Y_L matrix
    # Input: T_ul: DataFrame[n_id, id, label, feature (distances), right vector] - submatrix Tul containing used in equation
    # Input: l_broadcast: sc.broadcast(int) - no. of known labels
    # Input: u_broadcast: sc.broadcast(int) - no. of unknown labels
    # Output: Y: Dataframe[n_id, id, label, feature (distances), label_vector] - complete dataframe
    id = kwargs.get("id", "id")
    n_id = kwargs.get("n_id", "n_id")
    label = kwargs.get("label", "label")
    feature = kwargs.get("feature", "feature")
    spark = SparkSession(sparkContext=sc)

    identity_bm = create_eye(sc=sc, broadcast_u=u_broadcast,**kwargs)
    T_uu_bm = IndexedRowMatrix(T_uu.rdd.map(lambda x: IndexedRow(x[n_id], x["right"]))).toBlockMatrix()
    D_bm = identity_bm.subtract(T_uu_bm)

    # Convert D_bm back to DataFrame
    D_row = Row("vector", n_id)
    D_df = spark.createDataFrame(D_bm.toIndexedRowMatrix().rows.map(lambda x: D_row(x.vector, x.index)))
    # Compute (I-Tuu)^-1
    inverted_mat_df = invert(sc=sc, data_frame=D_df, column="vector", id_col=n_id)
    invD_bm = IndexedRowMatrix(inverted_mat_df.rdd.map(lambda x: IndexedRow(x[n_id], x["inverted_array"]))).toBlockMatrix()

    # Create Y_l as respectfully (u x C) matrices. C is the number of class'
    broad_c = sc.broadcast(T_ll.select(label).distinct().count())
    y_l_df = label_matrix(data_frame=T_ll, broadc_classes=broad_c, label_col=label)
    T_ul_bm = IndexedRowMatrix(T_ul.rdd.map(lambda x: IndexedRow(x[n_id], x["left"]))).toBlockMatrix()
    y_l_bm = IndexedRowMatrix(y_l_df.rdd.map(lambda x: IndexedRow(x[n_id], x["vector_labels"]))).toBlockMatrix()

    # Do the computation (I-T_uu)^-1T_ulY_l
    y_u_bm = invD_bm.multiply(T_ul_bm.multiply(y_l_bm))
    temp_y_u_df = y_u_bm.toIndexedRowMatrix().rows.map(lambda x: D_row(x.vector, x.index)).toDF()
    y_u_df = T_uu.select(id, n_id, label, feature).join(temp_y_u_df, on=n_id, how="inner")

    output = y_l_df.drop("left").union(y_u_df)
    return output

def create_eye(sc: SparkContext, broadcast_u, **kwargs):
    # Creates a Identity matrix as a BlockMatrix
    # Input: sc:SparkContext -
    # Input: broadcast_u:sc.broadcast(int) - broadcasted no. of row/cols
    # Input: kwargs:dict - containing additional arguments
    # Output: eye:BlockMatrix - identity block matrix
    no_rows_block = kwargs.get('no_rows_block', 1024)
    no_cols_block = kwargs.get('no_cols_block', 1024)
    return IndexedRowMatrix(sc.
                    range(broadcast_u.value).
                    map(lambda x: (x, Vectors.sparse(int(broadcast_u.value), [int(x)], [1.0]))).
                    map(lambda x: IndexedRow(*x))
                    ).toBlockMatrix(no_rows_block, no_cols_block)

