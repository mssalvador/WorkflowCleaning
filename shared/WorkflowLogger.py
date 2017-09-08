#Make this a class to use @ for logger so we can save alot of lines of code

from functools import partial


def _pseudo_def_log_info_decorator(orig_function, argument='/tmp/workflow_test.log', logger_type='info'):
    import logging

    def wrapper_func(*args, **kwargs):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger_file_handler = logging.FileHandler(
            filename=argument)
        logger_format = logging.Formatter(
            '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

        logger.addHandler(logger_file_handler)
        logger_file_handler.setFormatter(
            logger_format)
        if logger_type == 'info':
            logger.info('Ran {} with args {} and kwargs {}'
                        .format(orig_function.__name__, args, kwargs))
        elif logger_type == 'warning':
            logger.warning()
        elif logger_type == 'debug':
            logger.debug()
        else:
            logger.error()

        # print('wrapper ran before {}'
        #       .format(args))

        return orig_function(*args, **kwargs)
    return wrapper_func

def_logger_info = partial(_pseudo_def_log_info_decorator)
def_logger_warn = partial(_pseudo_def_log_info_decorator, logger_type='warning')
def_logger_debug = partial(_pseudo_def_log_info_decorator, logger_type='debug')
def_logger_err = partial(_pseudo_def_log_info_decorator, logger_type='error')









