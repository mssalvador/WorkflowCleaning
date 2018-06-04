from shared.WorkflowLogger import logger_info_decorator, logger
from ast import literal_eval


@logger_info_decorator
def parse_algorithm_variables(vars):
    """
    Parses variables that are supplied as strings
    :param vars: list of string variables
    :return: list of strings with variables in the correct Python format
    """
    for key, val in vars.items():
        try:
            vars[key] = literal_eval(val)
        except ValueError as ve:
            # print('Data {} is of type {}'.format(key, type(val)))
            logger.info('Data {} is of type {}'.format(key, type(val)))
        except SyntaxError as se:
            vars[key] = val.strip(' ')
            logger.error('Data {} is of type {}'.format(key, type(val)))
    return vars