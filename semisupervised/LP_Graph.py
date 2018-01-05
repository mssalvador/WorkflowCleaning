import numpy as np
from pyspark.mllib.linalg import distributed
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.linalg import SparseVector, VectorUDT
from shared import ConvertAllToVecToMl
from pyspark.ml import Pipeline
from shared.WorkflowLogger import logger_info_decorator


@logger_info_decorator
def _compute_weights(vec_x, vec_y, sigma):
    if isinstance(vec_y, SparseVector) | isinstance(vec_x, SparseVector):
        x_d = vec_x.toArray()
        y_d = vec_y.toArray()
        return float(np.exp(-(np.linalg.norm(x_d-y_d, ord=2)**2)/sigma**2))
    else:
        return float(np.exp(-np.linalg.norm(vec_x - vec_y, ord=2)**2 / sigma**2))


@logger_info_decorator
def create_complete_graph(sc, data_frame, feature_columns, id_column='id',
                          label_column='label', standardize=True, sigma=0.7):
    std_feature_name = 'standard_feat'
    feature_gen = VectorAssembler(inputCols=feature_columns, outputCol='features')
    converter = ConvertAllToVecToMl.ConvertAllToVecToMl(inputCol='features', outputCol='converteds')
    standardize = StandardScaler(withMean=standardize, withStd=standardize,
        inputCol=converter.getOutputCol(), outputCol=std_feature_name)
    pipeline = Pipeline(stages=[feature_gen, converter, standardize])
    model = pipeline.fit(data_frame)

    vector_df = model.transform(data_frame)
    to_sparse_udf = F.udf(lambda x: SparseVector(len(x), [(i,j) for i,j in enumerate(x) if j != 0]), VectorUDT())
    standard_X_sparse = vector_df.withColumn('weights', to_sparse_udf(F.col(std_feature_name)))
    bc_vec = sc.broadcast(standard_X_sparse.select(id_column,'weights').rdd.collectAsMap())

    rdd_srink = standard_X_sparse.rdd.map(lambda x: (x[id_column], x[label_column]))
    rdd_cartesian = (rdd_srink
                     .cartesian(rdd_srink)
                     .map(lambda x: (*x[0], *x[1]))
                     .map(lambda x: distributed.MatrixEntry(
                         x[0], x[2], _compute_weights(bc_vec.value.get(x[0]),
                                                      bc_vec.value.get(x[2]),
                                                      sigma=sigma)))
                    )
    return distributed.CoordinateMatrix(rdd_cartesian)