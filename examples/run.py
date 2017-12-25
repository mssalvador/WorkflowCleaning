import examples
from shared.parse_algorithm_variables import parse_algorithm_variables

default_lp_param = {'method': 'double_helix', 'n': 300,
            'alpha': 1.0, 'beta': 1.0, 'missing': 1}

def run(sc, **kwargs):
    # select the data
    example = parse_algorithm_variables(kwargs.get(
        'algo_params', default_lp_param))
    if example['method'] == 'double_helix':
        for key in default_lp_param.keys():
            if key not in example:
                example[key] = default_lp_param[key]
        label = kwargs.get('labels', 'unknown_label')
        examples.double_helix(sc=sc, example=example, label=label)
    elif example['method'] == 'mnist':
        train_df, test_df = examples.mnist(sc=sc, **example)
        if example['package'] == 'spark':
            train_df.show(5)
        return train_df