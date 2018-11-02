import pprint
import pyspark
from pyspark.sql import functions as F
from pyspark.mllib.linalg.distributed import MatrixEntry

from semisupervised.labelpropagation.lp_generate_graph import do_cartesian


def preamble(sc, input_data: pyspark.sql.DataFrame, id_col='id', **kwargs):
    # Bekskrivelse: Denne metode skal håndtere de inputs som kommer fra main, dvs data og parameter
    # input: **kwargs
    # output: data, og key/val args som er inputs til metoden

    feature_cols = kwargs.get('feature_cols', None)
    if not contains_right_features(feature_cols):
        raise Exception('Missing data elements in features {}'.format(", ".join(feature_cols)))

    # Data håndters
    if kwargs.get('datatype', 'raw') == 'preprocessed':
        n = 1+input_data.select(F.max('label_a').alias('max')).collect()[0]['max']
        if kwargs.get('squared', False):
            squared_data = input_data.withColumn(
                colName='squaredDistance',
                col=F.col('distance')**2
            )
        else:
            squared_data = input_data.withColumn(
                colName='squaredDistance',
                col=F.col('distance')
            )
        rdd_squared_data = squared_data.rdd.map(lambda x: MatrixEntry(
            i=x['label_a'],
            j=x['label_b'],
            value=x['squaredDistance']))

    else:  # Vi laver et cartesian produkt
        n = input_data.count()
        rdd_squared_data = (do_cartesian(
            sc=sc,
            df=input_data,
            id_col=id_col,
            feature_col=feature_cols,
            **kwargs
        ).persist(pyspark.StorageLevel(True, True, False, False)))
        print(rdd_squared_data.take(1))
    return rdd_squared_data, n


def contains_right_features(features):
    # Bekskrivelse: Undersøger om alle features i den præprocesseret udgave er der
    # input: features - list med strings
    # output: output - Boolean
    rigth_features = ('label_a', 'label_b', 'distance')
    for f in features:
        if f not in rigth_features:
            return False
    return True
