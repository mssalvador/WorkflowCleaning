from pyspark import SparkContext
from pyspark.sql import SparkSession
def run(sc: SparkContext, **kwargs):
    """
    :param sc:
    :param kwargs:
    :return:
    """
    # Import data
    spark = SparkSession(sparkContext=sc)


    # Execute model


    # Return result