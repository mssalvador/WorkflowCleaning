from shared.create_dummy_data import create_double_helix, load_mnist
from semisupervised.LabelPropagation import label_propagation
from shared.parse_algorithm_variables import parse_algorithm_variables
from shared.Plot2DGraphs import plot3D

def run(sc, **kwargs):

    # select the data
    example = parse_algorithm_variables(kwargs.get(
        'algo_params', {
            'method': 'double_helix', 'n': 300,
            'alpha': 1.0, 'beta': 1.0 , 'label': 'unknown_label' })
    )
    if example['method'] == 'double_helix':
        double_helix = create_double_helix(
            example['n'], alpha=example['alpha'], beta=example['beta'])
        plot3D(double_helix, **example)
    elif example['method'] == 'mnist':
        pdf = load_mnist()
        print(pdf)






