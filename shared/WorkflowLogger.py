# Make this a class to use @ for logger so we can save a lot of lines of code

from functools import partial, wraps
import logging
import sys
import datetime


def create_logger(argument='/tmp/workflow_test.log'):
    """
    creates a logging object that can be used later in different decorators
    :return: logging object
    """
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create handler
    logger_file_handler = logging.FileHandler(
        filename=argument)

    # create formatter
    logger_format = logging.Formatter(
        '%(asctime)s;%(levelname)s;%(name)s;%(message)s')

    # add logger_file_handler to logger
    logger.addHandler(logger_file_handler)

    # add logger_format to logger_file_handler
    logger_file_handler.setFormatter(
        logger_format)

    return logger

now = datetime.datetime.now()
str_now = str(now).replace(':','_').replace('-','_').replace('.','_')
logger = create_logger(
    argument='/tmp/{}_workflow_test.log'.format(str_now))


def _log_info(orig_function, logger=logger):

    @wraps(orig_function)
    def wrapper_func(*args, **kwargs):
        # 'application' code
        try:
            # print('Ran {} with args {} and kwargs {}'
            #       .format(orig_function.__name__, args, kwargs))
            logger.info('Ran {}; args {}; kwargs {}'
                        .format(orig_function.__name__,
                                args, kwargs)
                        )
            return orig_function(*args, **kwargs)
        except Exception as e:
            tb = sys.exc_info()[2]
            logger.error(e.with_traceback(tb))
            raise
    return wrapper_func


# Decorators to be used
logger_info_decorator = partial(_log_info, logger=logger)  # info
