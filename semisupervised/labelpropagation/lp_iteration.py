import labelpropagation


def propagation_step(transition_matrix, label_matrix, max_iterations=25):
    iterations = 0
    persisted_transition_matrix = transition_matrix.cache()
    persisted_transition_matrix.take(1)
    while iterations < max_iterations:
        new_y_matrix = labelpropagation.naive_multiplication_rdd(
            mat_a=persisted_transition_matrix, mat_b=label_matrix, is_triangle=True)
        iterations += 1
    return new_y_matrix
