# Make this a class to use @ for logger so we can save a lot of lines of code

from functools import partial
from functools import wraps

def _pseudo_def_log_info_decorator(orig_function, argument='/tmp/workflow_test.log', logger_type='info'):
    import logging

    @wraps(orig_function)
    def wrapper_func(*args, **kwargs):

        # create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create handler
        logger_file_handler = logging.FileHandler(
            filename=argument)

        # create formatter
        logger_format = logging.Formatter(
            '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

        # add logger_file_handler to logger
        logger.addHandler(logger_file_handler)

        # add logger_format to logger_file_handler
        logger_file_handler.setFormatter(
            logger_format)

        # 'application' code
        if logger_type == 'info':
            logger.info('Ran {} with args {} and kwargs {}'
                        .format(orig_function.__name__, args, kwargs))
            print('Ran {} with args {} and kwargs {}'
                  .format(orig_function.__name__, args, kwargs))
        elif logger_type == 'warning':
            logger.warning('Ran {} with args {} and kwargs {}'
                           .format(orig_function.__name__, args, kwargs))
            print('Ran {} with args {} and kwargs {}'
                  .format(orig_function.__name__, args, kwargs))
        elif logger_type == 'debug':
            logger.debug('Ran {} with args {} and kwargs {}'
                         .format(orig_function.__name__, args, kwargs))
            print('Ran {} with args {} and kwargs {}'
                  .format(orig_function.__name__, args, kwargs))
        else:
            logger.error('Method {} failed with args {} and kwargs {}'
                         .format(orig_function.__name__, args, kwargs))
            print('Ran {} with args {} and kwargs {}'
                  .format(orig_function.__name__, args, kwargs))

        # print('wrapper ran before {}'
        #       .format(args))

        return orig_function(*args, **kwargs)
    return wrapper_func


# Decorators to be used
def_logger_info = partial(_pseudo_def_log_info_decorator)  # info
def_logger_warn = partial(_pseudo_def_log_info_decorator, logger_type='warning')
def_logger_debug = partial(_pseudo_def_log_info_decorator, logger_type='debug')
def_logger_err = partial(_pseudo_def_log_info_decorator, logger_type='error')









