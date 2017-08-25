"""
This contains the left overs from python 3.5.2 which are not contained in python 3.5.3

"""

import math
import sys
from IPython.core.magics.execution import TimeitResult


def average(l):
    return math.fsum(l) / len(l)


def stdev(l):
    mean = average(l)
    return (math.fsum([(x - mean) ** 2 for x in l]) / len(l)) ** 0.5

def print_human_readable_time(timespan, precision=3):
    """
    Gets a converted time
    @param: timespan: time in seconds
    @param: precision: the precision of time
    """
    if timespan >= 60.0:
        parts = [("d", 60 * 60 * 24), ("h", 60 * 60), ("min", 60), ("s", 1)]
        time = []
        leftover = timespan
        for suffix, length in parts:
            val = int(leftover / length)
            if val > 0:
                leftover = leftover % length
                time.append(u'{!s}{!s}'.format(str(val), suffix))
        return " ".join(time)

    units = [u"s", u"ms", u'us', "ns"]  # the save value
    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb5'.encode(sys.stdout.encoding)
            units = [u"s", u"ms", u'\xb5s', "ns"]
        except:
            pass
    scaling = [1, 1e3, 1e6, 1e9]

    if timespan > 0.0:
        order = min(-int(math.floor(math.log10(timespan)) // 3), 3)
    else:
        order = 3
    return u"%.*g %s" % (precision, timespan * scaling[order], units[order])


def pretty_time_result(timer):
    assert isinstance(timer, TimeitResult), "timer is wrong"

    pm = '+-'
    timings = [dt / timer.loops for dt in timer.all_runs]


    if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
        try:
            u'\xb1'.encode(sys.stdout.encoding)
            pm = u'\xb1'
        finally:
            return (
                u"{mean} {pm} {std} per loop (mean {pm} std. dev. of {runs} run{run_plural}, {loops} loop{loop_plural} each)"
                    .format(
                    pm=pm,
                    runs=timer.repeat,
                    loops=timer.loops,
                    loop_plural="" if timer.loops == 1 else "s",
                    run_plural="" if timer.repeat == 1 else "s",
                    mean=print_human_readable_time(average(timings), timer._precision),
                    std=print_human_readable_time(stdev(timings), timer._precision)
                )
            )