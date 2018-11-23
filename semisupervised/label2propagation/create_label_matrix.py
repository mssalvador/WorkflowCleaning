from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import VectorUDT

def label_matrix(data_frame: DataFrame, broadc_classes, label_col):
    # creates a new label matrix Y of dimension data_frame_len x classes
    create_class_vector = F.udf(lambda x: determine_class(x, broadc_classes.value), VectorUDT())
    return data_frame.withColumn("vector_labels", create_class_vector(label_col))


def determine_class(cls_value, classes):
    if cls_value is float("nan"):
        return Vectors.dense([0.0]*classes)
    else:
        return Vectors.dense([1.0 if idx == cls_value-1 else 0.0 for idx in range(classes)])