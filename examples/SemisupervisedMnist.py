import shared.create_dummy_data as ccd
import semisupervised as ss
import pyspark

def mnist(sc : pyspark.SparkContext, **kwargs):
    # This will be the mnist performance test module
    spark_session = pyspark.sql.SparkSession(sparkContext=sc)
    train_pdf, test_pdf = ccd.load_mnist(sc, **kwargs)
    # train_df = spark_session.createDataFrame(train_pdf)
    # test_df = spark_session.createDataFrame(test_pdf)
    # ss.label_propagation(sc=sc, data_frame=train_df,)
    return train_pdf, test_pdf