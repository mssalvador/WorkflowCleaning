from pyspark.mllib.linalg import distributed
import labelpropagation


def label_propagation(sc, data_frame=None, id_col='id', label_col='label', feature_cols=None, **kwargs):
    """
    New Version of Labelpropagation with sparks matrix lib used
    :param sc:
    :param data_frame:
    :param id_col:
    :param label_col:
    :param feature_cols:
    :param kwargs: iterations, tol, standardize, sigma, priors, evaluation_type, k
    :return:
    """
    n = data_frame.count()
    cartesian_demon_rdd = labelpropagation.do_cartesian(
        sc=sc, df=data_frame, id_col=id_col, feature_col=feature_cols, **kwargs).cache()

    demon_matrix = distributed.CoordinateMatrix(entries=cartesian_demon_rdd, numRows=n, numCols=n)
    row_summed_matrix = demon_matrix.entries.flatMap(labelpropagation.lp_helper.triangle_mat_summation).reduceByKey(
        lambda x, y: x + y).collectAsMap()
    bc_row_summed = sc.broadcast(row_summed_matrix)
    # print(type(bc_row_summed.value))

    transition_rdd = demon_matrix.entries.map(
        lambda x: distributed.MatrixEntry(
            i=x.i, j=x.j, value=x.value / bc_row_summed.value.get(x.j))
    )
    col_summed_matrix = (transition_rdd.flatMap(labelpropagation.lp_helper.triangle_mat_summation)
        .reduceByKey(lambda x, y: x + y).collectAsMap())
    bc_col_summed = sc.broadcast(col_summed_matrix)

    hat_transition_rdd = transition_rdd.map(
        lambda x: distributed.MatrixEntry(
            i=x.i, j=x.j, value=x.value / bc_col_summed.value.get(x.i))
    )

    initial_y_matrix = labelpropagation.lp_helper.generate_label_matrix(
        df=data_frame, label_col=label_col, id_col=id_col, k=kwargs.get('k', None))
    final_label_matrix = labelpropagation.propagation_step(
        transition_matrix=hat_transition_rdd, label_matrix=initial_y_matrix,
        max_iterations=kwargs.get('max_iters', 25))
    coordinate_label_matrix = distributed.CoordinateMatrix(
        entries=final_label_matrix, numRows=initial_y_matrix.numRows(),
        numCols=initial_y_matrix.numCols())

    output_data_frame = labelpropagation.lp_helper.merge_data_with_label(
        sc=sc, org_data_frame=data_frame, coordinate_label_rdd=coordinate_label_matrix, id_col=id_col)

    return labelpropagation.lp_helper.evaluate_label_based_on_eval(
        sc=sc, data_frame=output_data_frame, label_col=label_col, **kwargs)
