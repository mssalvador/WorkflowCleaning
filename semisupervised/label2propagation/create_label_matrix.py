from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import VectorUDT

import math

def label_matrix(data_frame: DataFrame, broadc_classes, label_col):
    # Creates a new label matrix Y of dimension data_frame_len x classes
    # Input: data_frame[id:int, label:int, features:vector] - containing distance values for points
    # Input: broadc_classes:sc.broadcast - spark broadcast variable containing number of class'.
    # Input: label_col:str - name of label column in data_frame
    # Ouput: data_frame[id:int, label:int, features:vector, vector_labels:vector
    create_class_vector = F.udf(lambda x: determine_class(x, broadc_classes.value), VectorUDT())
    return data_frame.withColumn("vector_labels", create_class_vector(label_col))


def determine_class(cls_value, classes):
    # Constructs a vector of no. class' long containing either 1.0 on label index - 1 or all zeros
    # Input: cls_value: float - value of data points class label
    # Input: classes: int - no. class'
    # Output: Vectors.dense - dense spark vector containing the class'
    if math.isnan(cls_value):
        return Vectors.dense([0.0]*classes)
    else:
        return Vectors.dense([1.0 if idx == int(cls_value-1) else 0.0 for idx in range(classes)])