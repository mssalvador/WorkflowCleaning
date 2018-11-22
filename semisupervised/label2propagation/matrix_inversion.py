from pyspark import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import SQLContext

def invert(sc: SparkContext, data_frame: DataFrame, column="IminusD", output_cols="inverted_array", id_col="id"):
    sqlCtx = SQLContext.getOrCreate(sc=sc)
    inverted_df =  DataFrame(sc._jvm.com.github.mssalvador.matrixInv.MatrixInv.loadData(
        sc._jsc.sc(),
        data_frame._jdf,
        column),
        sql_ctx=sqlCtx
    )

    return (inverted_df.
            withColumnRenamed(existing="_2", new=id_col).
            withColumnRenamed(existing="_1", new=output_cols).
            join(data_frame, on=id_col, how="inner").
            sort(id_col)
            )