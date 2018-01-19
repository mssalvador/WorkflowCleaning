import labelpropagation
import itertools


def propagation_step(sc, transition_matrix, label_matrix, clamped=None, max_iterations=25):
    """
    Does max_iterations mutliplications.
    :param transition_matrix:
    :param label_matrix:
    :param max_iterations:
    :param clamped: array with clamped
    :return:
    """
    iterations = 1
    persisted_transition_matrix = transition_matrix.cache()
    persisted_transition_matrix.take(1)
    clamped_rows = clamped.map(lambda x: (x.i, x.j)).collect()
    clamped_dict = _generate_clamped_zeros(clamped_rows, label_matrix.numCols())
    clamped_rdd = clamped.map(lambda x: ((x.i, x.j), 1.0))
    try:
        clamped_dict.update(clamped_rdd.collectAsMap())
    except Exception as e:
        print('Clamped_rdd is {}'.format(clamped_rdd.collectAsMap()))
    broadcasted_clamped = sc.broadcast(clamped_dict)

    new_y_matrix = labelpropagation.naive_multiplication_rdd(
        mat_a=persisted_transition_matrix, mat_b=label_matrix, is_triangle=True)
    new_y_matrix =_remove_clamped_values(
        label_matrix=new_y_matrix, clamped=clamped, broad_casted_clamped=broadcasted_clamped)

    while iterations < max_iterations:
        new_y_matrix = labelpropagation.naive_multiplication_rdd(
            mat_a=persisted_transition_matrix, mat_b=new_y_matrix, is_triangle=True)
        new_y_matrix = _remove_clamped_values(
            label_matrix=new_y_matrix, clamped=clamped, broad_casted_clamped=broadcasted_clamped)
        iterations += 1
        # print(new_y_matrix.take(10))
    return new_y_matrix

def _generate_clamped_zeros(clamped_values, n_label_cols):
    columns = range(n_label_cols)
    return dict(itertools.chain(*[[((r[0], i), 0.0) for i in columns if i != r[1]] for r in clamped_values]))

def _remove_clamped_values(label_matrix, clamped, broad_casted_clamped):
    return (label_matrix
        .filter(lambda x: (x.i, x.j) not in broad_casted_clamped.value)
        .union(clamped))
