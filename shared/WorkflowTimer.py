import time
import functools


def timeit(repeat=3):
    def time_method(func):
        @functools.wraps(func)
        def wrapper_func(*args, **kwargs):
            # 'application' code
            timer_list = []
            _i = 0
            while True:
                _t0 = time.time()
                retval = func(*args, **kwargs)
                _t1 = time.time()
                timer_list.append(_t1-_t0)
                if _i >= repeat:
                    break
                _i += 1
            return timer_list, retval
        return wrapper_func
    return time_method


@timeit(5)
def dummy_func(value):
    return value+1231