import numpy as np
from pyspark.sql import types as T
from pyspark.mllib import linalg
from pyspark.sql import functions as F
from shared import context
import math
import sys
from functools import partial, reduce
from shared.WorkflowLogger import logger_info_decorator, logger
from semisupervised.LP_Graph import create_complete_graph
from semisupervised.ClassMassNormalisation import class_mass_normalization


@logger_info_decorator
def label_propagation(
        sc, data_frame, label_col='label', id_col='id', feature_cols=None,
        k=2, sigma=0.7, max_iters=5, tol=0.05, standardize=True,
        eval_type='max', priors=None):
    """
    The actual label propagation algorithm
    :param:eval_type: 'Type of after evaluation; two are supported
    max: maximum likelihood
    cmn: Class mass normalisation
    """
    # initial stuff sets up a jobcontext for shared values here.
    label_context = context.JobContext(sc)
    label_context_set = partial(label_context.set_constant, sc)
    label_context_set('k', k)
    label_context_set('n', data_frame.count())
    label_context_set('tol', tol)
    label_context_set('priors', priors)

    # Lets make a proper dataset
    graph_matrix = create_complete_graph(
        data_frame=data_frame, feature_columns=feature_cols, id_column=id_col,
        label_column=label_col, sigma=sigma, standardize=standardize)

    one_matrix = linalg.Matrices.dense(1, data_frame.count(), np.ones(data_frame.count()))
    blocks1 = sc.parallelize([((0, 0), one_matrix)])
    one_block_matrix = linalg.distributed.BlockMatrix(blocks1, 1024, 1024)

    row_summation_matrix = one_block_matrix.multiply(graph_matrix)