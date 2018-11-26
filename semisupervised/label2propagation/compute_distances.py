from pyspark.sql import DataFrame
from pyspark import Row
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.mllib.linalg import DenseVector, VectorUDT
from pyspark import SparkContext
import numpy as np
import itertools as it

SIGMA = 0.3

def compute_distances(sc: SparkContext, data_frame: DataFrame, **kwargs):
    # Computes the distances between all points
    # Input: sc - SparkContext
    # Input: data_frame:DataFrame[id, label, feature] -
    # Input: kwargs:dict - containing all parameters
    # Output: output:DataFrame[id, label, distances:vector]
    id_col = kwargs.get("id", "id")
    label_col = kwargs.get("label", "label")
    feature_col = kwargs.get("feature", "feature")
    cnt = data_frame.count()
    indicies_rdd = sc.parallelize(list(it.combinations_with_replacement(range(cnt), 2)))
    all_vectors_df = (indicies_rdd.
                      toDF(["id", "jd"]).
                      join(data_frame, on="id", how="inner").
                      withColumnRenamed(feature_col,"i_"+feature_col).
                      withColumnRenamed(label_col, "i_"+label_col)
                      )
    # all_vectors_df.show()
    vectors_df = (data_frame.
                  join(all_vectors_df, all_vectors_df.jd==data_frame.id, how="inner").
                  drop(data_frame.id).
                  withColumnRenamed(feature_col, "j_"+feature_col).
                  withColumnRenamed(label_col, "j_" + label_col)
                  )
    squared_dist = F.udf(lambda x, y: float(np.linalg.norm(x-y)**2/SIGMA), T.DoubleType())

    output_df = (vectors_df.
                 select(id_col, "jd", "i_"+label_col, "j_"+label_col, squared_dist("i_"+feature_col,"j_"+feature_col).alias("w")).
                 sort(id_col, "jd").
                 rdd.
                 flatMap(lambda x: create_opposite_mat_elem(x=x, idx=id_col, jdx="jd", ilabel="i_"+label_col, jlabel="j_"+label_col, val="w")).
                 toDF().
                 groupBy("id", "label").
                 agg(F.collect_list(F.struct(F.col("jd"), F.col("w"))).alias("vector")).
                 rdd.
                 map(lambda row: Row(id=row["id"], label=row["label"], vector=create_distance_vector(row["vector"]))).
                 toDF()
                 # sort("i_"+label_col)
                 )
    return output_df


def create_distance_vector(x):
    # Converts a list of lists into a vector. Converts a lower triangle matrix into a full matrix with 0 in diag.
    # Input: x - list[list(index:int, value:double)]
    # Inpit: length - sc.broadcast value, containing length of the vectors
    # Output: output: - mllib.linalg.DenseVector, containing distances from: i to j.
    vector = [i[1] for i in sorted(x, key=take_first)]
    return DenseVector(vector)

def take_first(elem):
    return elem[0]

def create_opposite_mat_elem(x, **kwargs):
    # Take a tuple of (i,j,value) and dupplicates it
    # Input: x - tuple(i, j, value) row i, column j value
    # Output: value - duplicated element
    i = kwargs.get("idx", "id")
    j = kwargs.get("jdx", "jd")
    ilab = kwargs.get("ilabel", "ilabel")
    jlab = kwargs.get("jlabel", "jlabel")
    val = kwargs.get("val", "w")
    if x[i] == x[j]:
        return [Row(id=x[i], jd=x[j], label=x[ilab], w=x[val])]

    return [Row(id=x[i], jd=x[j], label=x[ilab], w=x[val]),
            Row(id=x[j], jd=x[i], label=x[jlab], w=x[val])]