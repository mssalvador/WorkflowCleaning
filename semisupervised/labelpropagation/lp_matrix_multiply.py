from pyspark import RDD
from pyspark.mllib.linalg import distributed


def coordinate_matrix_decorate(func):
    def func_wrapper(mat_a, mat_b, is_triangle=False):
        if isinstance(mat_a, distributed.CoordinateMatrix):
            mat_a = mat_a.entries
        if isinstance(mat_b, distributed.CoordinateMatrix):
            mat_b = mat_b.entries
        return func(mat_a, mat_b, is_triangle)
    return func_wrapper


@coordinate_matrix_decorate
def naive_multiplication_rdd(mat_a: RDD, mat_b: RDD, is_triangle=False):
    """
    mat_a is the left matrix
    mat_b is the right matix
    :param mat_a:
    :param mat_b:
    :param is_triangle:
    :return:
    """
    if is_triangle:
        left_rdd = (mat_a.
                    flatMap(lambda x: [((x.j, x.i), x.value), ((x.i, x.j), x.value)]).
                    aggregateByKey(zeroValue=(0.0, 0.0),
                                   seqFunc=lambda x, y: (x[0]+y, x[1]+1),
                                   combFunc=lambda x, y: (x[0] + y[0], x[1]+y[1])
                                   ).
                    mapValues(lambda x: x[0] / x[1]).
                    map(lambda x: (x[0][0], (x[0][1], x[1])))
                    )
    else:
        left_rdd = mat_a.map(lambda x: (x.j, (x.i, x.value)))

    right_rdd = mat_b.map(lambda x: (x.i, (x.j, x.value)))
    combined_rdd = (left_rdd.join(right_rdd).map(lambda x: x[1]).
                    map(lambda x: ((x[0][0], x[1][0]), x[0][1]*x[1][1])).
                    reduceByKey(lambda x, y: x+y).
                    map(lambda x: distributed.MatrixEntry(i=x[0][0], j=x[0][1], value=x[1]))
                    )
    return combined_rdd
