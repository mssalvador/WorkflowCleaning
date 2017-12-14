import examples
from shared.parse_algorithm_variables import parse_algorithm_variables

default_lp_param = {'method': 'double_helix', 'n': 300,
            'alpha': 1.0, 'beta': 1.0, 'missing': 1}

def run(sc, **kwargs):
    # select the data
    example = parse_algorithm_variables(kwargs.get(
        'algo_params', default_lp_param))
    for key in default_lp_param.keys():
        if key not in example:
            example[key] = default_lp_param[key]

    label =  kwargs.get('labels', 'unknown_label')
    if example['method'] == 'double_helix':
        examples.double_helix(sc=sc, example=example, label=label)

    elif example['method'] == 'mnist':
        train_pdf, test_pdf = examples.mnist()
        print(train_pdf)