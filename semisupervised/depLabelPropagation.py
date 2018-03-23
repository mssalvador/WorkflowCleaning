class label_propagation2(object):
    def __init__(self, sc, id_col='id', label_col='label', feature_cols=None, **kwargs):
        self.sc = sc
        self.id_col = id_col
        self.label_col = label_col
        self.feature_cols = feature_cols
        self.kwargs = kwargs

    def run(self, data_frame):
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
        #from semisupervised.labelpropagation.lp_helper import triangle_mat_summation
        from semisupervised.labelpropagation.lp_helper import generate_label_matrix
        from semisupervised.labelpropagation.lp_helper import merge_data_with_label
        from semisupervised.labelpropagation.lp_helper import evaluate_label_based_on_eval

        def triangle_mat_summation(mat_element: distributed.MatrixEntry):
            if mat_element.j == mat_element.i:
                return (mat_element.i, mat_element.value),
            else:
                return (mat_element.i, mat_element.value), (mat_element.j, mat_element.value)

        n = data_frame.count()
        max_iter = self.kwargs.get('max_iters', 25)
        cartesian_demon_rdd = (do_cartesian(
            sc=self.sc, df=data_frame, id_col=self.id_col, feature_cols=self.feature_cols, **self.kwargs)
            .persist(StorageLevel(True, True, False, False))
        )
        cartesian_demon_rdd.take(1)

        demon_matrix = distributed.CoordinateMatrix(entries=cartesian_demon_rdd, numRows=n, numCols=n)
        row_summed_matrix = (demon_matrix.entries.flatMap(triangle_mat_summation))#.collectAsMap())
        print(row_summed_matrix.take(5))
        time.sleep(10)
        bc_row_summed = self.sc.broadcast(row_summed_matrix.reduceByKey(lambda x, y: x + y).collectAsMap())
        # print(type(bc_row_summed.value))

        transition_rdd = demon_matrix.entries.map(
            lambda x: distributed.MatrixEntry(
                i=x.i, j=x.j, value=x.value / bc_row_summed.value.get(x.j))
        )
        col_summed_matrix = (transition_rdd.flatMap(triangle_mat_summation)
            .reduceByKey(lambda x, y: x + y).collectAsMap())
        bc_col_summed = self.sc.broadcast(col_summed_matrix)

        hat_transition_rdd = transition_rdd.map(
            lambda x: distributed.MatrixEntry(
                i=x.i, j=x.j, value=x.value / bc_col_summed.value.get(x.i))
        ).persist()
        hat_transition_rdd.take(1)
        # cartesian_demon_rdd.unpersist() # Memory Cleanup!

        clamped_y_rdd, initial_y_matrix = generate_label_matrix(
            df=data_frame, label_col=self.label_col, id_col=self.id_col, k=self.kwargs.get('k', None))

        final_label_matrix = lp_iteration.propagation_step(
            self.sc, transition_matrix=hat_transition_rdd, label_matrix=initial_y_matrix,
            clamped=clamped_y_rdd, max_iterations=max_iter, )

        coordinate_label_matrix = distributed.CoordinateMatrix(
            entries=final_label_matrix, numRows=initial_y_matrix.numRows(),
            numCols=initial_y_matrix.numCols())

        output_data_frame = merge_data_with_label(
            sc=self.sc, org_data_frame=data_frame, coordinate_label_rdd=coordinate_label_matrix, id_col=self.id_col)

        hat_transition_rdd.unpersist() # Memory Cleanup!
        cartesian_demon_rdd.unpersist() # Memory Cleanup!
        return evaluate_label_based_on_eval(
            sc=self.sc, data_frame=output_data_frame, label_col=self.label_col, **self.kwargs)

