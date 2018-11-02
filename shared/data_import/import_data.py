import os
import re
import pyspark


def import_dataframe(spark_context, data, *args, **kwargs):
    # Bekskrivelse: Denne metode skal importere en csv fil til et spark dataframe
    # input: spark_context, data
    # output: data_frame

    if not isinstance(spark_context, pyspark.sql.SparkSession):
        raise Exception('spark context is not there. Got {}'.format(str(spark_context)))
    if os.path.exists(data):
        file_ending = re.findall('\.(csv|json|txt)', data)[-1]
        if file_ending == "txt":
            file_ending = "csv"
        data_frame = spark_context.read.format(file_ending).load(
            data,
            mode='PERMISSIVE',
            nullValue=float('NAN'),
            nanValue=float('NAN'),
            header=True,
            inferSchema=True)
    else:
        raise FileNotFoundError('path not found error')
    return data_frame
