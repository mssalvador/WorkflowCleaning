#import py4j modules
import py4j.protocol
from py4j.protocol import Py4JJavaError
from py4j.java_gateway import JavaObject
from py4j.java_collections import JavaArray, JavaList

#import spark modules
from pyspark import RDD
from pyspark.serializers import PickleSerializer, AutoBatchedSerializer
from shared.create_dummy_data import create_dummy_data
from pyspark import SparkContext

sc = SparkContext.getOrCreate()

# helper function to convert python objects to Jaba objects
def _to_java_object_rdd(rdd):
    """
    Return a JavaRDD of Object by unpickling.
    It will convert each Python object into Java object by Pyrolite, whenever the RDD
    is serialized in batch or not.
    """

    rdd = rdd._reserialize(AutoBatchedSerializer(PickleSerializer()))

    return rdd.ctx._jvm.org.apache.spark.mllib.api.python.SerDe.pythonToJava(rdd._jrdd, True)


def compute_size_of_dataframe(df):
    """
    Computes the size of a data frame by converting it into a javaRDD
    :param df:
    :return:
    """

    #First convert it to RDD
    java_obj = _to_java_object_rdd(df.rdd)
    bts = sc._jvm.org.apache.spark.util.SizeEstimator.estimate(java_obj)
    mb = bts/1000000
    print(str(mb)+"MB")
