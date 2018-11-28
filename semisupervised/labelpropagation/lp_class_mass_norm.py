from pyspark.sql import types as T
from pyspark.sql import functions as F
from functools import reduce


def _class_mass_calculation(label, priors):
    return map(lambda x: x[0]*x[1], zip(label, priors))


def _class_mass_parse_type(dataframe):
    t_count = dataframe.filter('is_clamped').count()
    print(t_count)
    prior_calc = F.count('label') / F.col('t_count')

    priors = (dataframe.
              filter('is_clamped').
              withColumn(colName='t_count', col=F.lit(t_count)).
              groupby('label', 't_count').
              agg(prior_calc).
              drop('t_count').
              rdd.
              collectAsMap()
              )
    print(priors)
    return list(map(
        func=lambda x: x[1],
        iter1=sorted(iterable=priors.items(),
                     key=lambda x: x[0]))
    )


def class_mass_normalization(context, data_frame):
    k = context.constants['k'].value
    priors = context.constants['priors'].value
    if isinstance(priors, list) and len(priors) == k:
        assert float(reduce(lambda x, y: x+y, priors)) == 1.0, "sum must be equal to 1.0, not {}".format(priors)
        priors = priors
    else:
        priors = _class_mass_parse_type(data_frame)
    udf_mass_calc = F.udf(
        f=lambda label: list(_class_mass_calculation(label, priors)),
        returnType=T.ArrayType(T.DoubleType())
    )
    return data_frame.withColumn(
        colName='initial_label',
        col=udf_mass_calc('initial_label')
    )
