#Make this a class to use @ for logger so we can save alot of lines of code

from functools import partial


def decorator_func(orig_function):
    def wrapper_func():
        print('wrapper executed before {}'.format(orig_function.__name__))
        return orig_function
    return wrapper_func()


def _pseudo_log_decorator(orig_function, argument):
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

        logger.info('Ran {} with args {} and kwargs {}'
                    .format(orig_function.__name__, args, kwargs))

        print('wrapper ran before {}'
              .format(orig_function.__name__))

        return orig_function(*args, **kwargs)
    return wrapper_func

decorator_logger_info = partial(_pseudo_log_decorator, argument='/tmp/workflow_test.log')

@decorator_logger_info
def display():
    print('display ran')








