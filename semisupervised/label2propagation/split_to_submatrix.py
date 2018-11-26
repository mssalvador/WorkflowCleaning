from pyspark.sql import dataframe
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import VectorUDT

def to_submatries(df: dataframe, broadcast_l, **kwargs):
    # to_subtrices splits the dataframe into 2 bottom sub dataframe each containing a "matrix" Rows of vectors
    # input: df - dataframe[id, features, label]
    # input: broadcast_l - broadcasted value containing no. known labels
    # input: kwargs - dict containing feature_col, id_col, etc
    # output: output - list(dataframe[id, feature, label] for submatrix)

    input_features = kwargs.get("features", "vectors")
    output_id = kwargs.get("output_id", "n_id")
    idx = kwargs.get("id", "id")
    label = kwargs.get("label", "label")
    split_schema = T.StructType([T.StructField("left", VectorUDT()), T.StructField("right", VectorUDT())])
    split_udf = F.udf(f=lambda x: [Vectors.dense(x[:broadcast_l.value]), Vectors.dense(x[broadcast_l.value:])],
                      returnType=split_schema) # Does the splitting of the dense vector into two dense vectors

    vert_splited_df = df.withColumn("splitted_vectors", split_udf(input_features)).cache()
    bottom_splitted_df = (vert_splited_df.
                          filter(F.isnan(label) | F.isnull(label)).
                          rdd.
                          zipWithIndex().
                          map(lambda x: [*x[0], x[1]]).toDF(vert_splited_df.columns+[output_id]))

    top_splitted_df = (vert_splited_df.
                          filter(~(F.isnan(label) | F.isnull(label))).
                          rdd.
                          zipWithIndex().
                          map(lambda x: [*x[0], x[1]]).toDF(vert_splited_df.columns+[output_id]))

    T_ll = top_splitted_df.select([output_id, idx]+[label]+[input_features]+["splitted_vectors.left"])
    T_lu = top_splitted_df.select([output_id, idx]+[label]+[input_features]+["splitted_vectors.right"])
    T_ul = bottom_splitted_df.select([output_id, idx]+[label]+[input_features]+["splitted_vectors.left"])
    T_uu = bottom_splitted_df.select([output_id, idx]+[label]+[input_features]+["splitted_vectors.right"])
    return [T_ll, T_lu, T_ul, T_uu]
