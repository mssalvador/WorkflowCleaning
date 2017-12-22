import shared.create_dummy_data as ccd
import semisupervised as ss
import pyspark

def mnist(sc : pyspark.SparkContext, **kwargs):
    # This will be the mnist performance test module
    spark_session = pyspark.sql.SparkSession(sparkContext=sc)
    train_data_frame, test_data_frame = ccd.load_mnist(sc, **kwargs)
    train_data_frame = train_data_frame.
    #ss.label_propagation(sc=sc, data_frame=train_df,)
    return train_data_frame, test_data_frame