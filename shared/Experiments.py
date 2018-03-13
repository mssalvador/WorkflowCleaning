from shared.WorkflowLogger import logger_info_decorator, logger
from shared.WorkflowTimer import timeit
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
import numpy as np


class Experiments(object):
    @logger_info_decorator
    def __init__(self, n_trials=None, data_size=None):
        if not data_size:
            self.data_size = [1000]
        else:
            self.data_size = data_size
        self.n_trials = n_trials
        self.execution_times = []

    def __str__(self):
        return '{}'.format('hat')

    def run_experiment(
            self, sc, data=None, functions=None,
            known_fraction=0.1,  **kwargs):
        feature_cols = kwargs.pop('feature_cols', None)
        label_col = kwargs.pop('label_col', None)
        output = None

        for d in self.data_size:
            sized_df = self.enlarge_dataset(
                dataframe=data, size=d, feature_cols=feature_cols,
                label_col=label_col, **kwargs
            )
            # Make dataset with nan values
            added_nan_df = Experiments.create_nan_labels(
                sc=sc, dataframe=sized_df, label_col=label_col,
                fraction=known_fraction, **kwargs
            )
            timer, output = self._execute_function(
                sc, func=functions, data=added_nan_df, **kwargs
            )
            # output.show()
            error_rate = Experiments._compute_error_rate(
                data_frame=output, original_label_col='missing_'+label_col,
                new_label_col='new_'+label_col
            )
            self.execution_times.append(
                ((d, known_fraction), (timer, error_rate))
            )
            Experiments.print_stats_time(timer, error_rate)
        return output

    @staticmethod
    def _compute_error_rate(data_frame, original_label_col, new_label_col):
        n = float(data_frame.count())
        new_label_equal_org_label = (F.col(original_label_col) == F.col(new_label_col))
        try:
            error_df = (data_frame
                .withColumn(colName='error',
                            col=F.when(new_label_equal_org_label, 0.).otherwise(1.))
                .groupBy()
                .agg(F.sum(F.col('error')).alias('error_rate'))
            )
            first_err_rate = (error_df
                .withColumn(colName='error_rate',
                            col=F.col('error_rate') / n)
                .collect()
            )
            return first_err_rate[0]['error_rate']
        except Exception as e:
            print(e)
            if original_label_col not in data_frame.columns:
                print('missing following column {}'
                      .format(original_label_col)
                      )
            if new_label_col not in data_frame.columns:
                print('missing following column {}'
                      .format(new_label_col)
                      )
            print([i for i in data_frame.columns if 'pixel' not in i])

    @timeit(5)
    def _execute_function(self, sc, func, data, **kwargs):
        return func(sc, data, **kwargs)

    @staticmethod
    def subset_dataset_by_label(sc, dataframe: DataFrame, label_col, *args):
        if args:
            return dataframe.filter(
                condition=F.col(label_col).isin(list(args))
            )
        else:
            return dataframe

    @staticmethod
    @logger_info_decorator
    def enlarge_dataset(dataframe: DataFrame, size=None,
                        feature_cols=None, label_col=None, **kwargs):
        # size must be larger than dataframe size
        n = dataframe.count()
        if n <= size:
            extras = (size - n) / n
            extra_df = dataframe.sample(
                withReplacement=True, fraction=extras
            )
            columns = [F.col(i) if i not in feature_cols
                       else F.col(i) + F.rand()
                       for i in extra_df.columns
                       ]
            return extra_df.select(columns).union(dataframe)
        else:
            frac = 1 - (n - size) / n
            samples = dataframe.sample(
                withReplacement=False, fraction=frac
            )
            all_types = list(map(
                func=lambda x: x[label_col],
                iter1=samples.select(label_col).distinct().collect())
            )
            while not (len(list(all_types)) == kwargs.get('k', 10)):
                print('iteration')
                samples = dataframe.sample(
                    withReplacement=False, fraction=frac
                )
                all_types = list(map(
                    func=lambda x: x[label_col],
                    iter1=samples.select(label_col).distinct().collect())
                )
            return samples

    @staticmethod
    def create_nan_labels(sc, dataframe, label_col, fraction=None, **kwargs):
        """
        Generates a column with either a missing factor for each label
        or n missing labels pr. label
        :param sc:
        :param dataframe:
        :param label_col:
        :param args:
        :return:
        """
        spark = SparkSession(sparkContext=sc)
        n = dataframe.count()
        dict_missing_labels = Experiments._compute_fraction(
            sc=sc, dataframe=dataframe, label_col=label_col, fraction=fraction, **kwargs)
        # print(dict_missing_labels)
        # create the new schema
        schema = dataframe.schema.add('id', data_type=T.IntegerType(), nullable=False)

        # add id to dataframe
        rdd = dataframe.rdd.zipWithIndex().map(lambda x: (*x[0], x[1]))
        new_data_frame = spark.createDataFrame(rdd, schema=schema)
        labeled_id = new_data_frame.sampleBy(label_col, fractions=dict_missing_labels).select('id').collect()

        bc_labels = sc.broadcast(list(map(lambda x: x['id'], labeled_id)))
        correct_label = F.when(
            condition=F.col('id').isin(bc_labels.value),
            value=F.col(label_col)).otherwise(float('NAN'))

        return new_data_frame.withColumn(colName='missing_' + label_col, col=correct_label)

    @staticmethod
    def _compute_fraction(sc, dataframe, label_col, fraction=None, **kwargs):
        if fraction:  # set all labels to same fraction
            bcast = sc.broadcast(fraction)
            dict_missing_labels = (dataframe
                .groupBy(label_col).count().rdd
                .map(lambda x: (x['label'], bcast.value)).collectAsMap())
        else:
            dict_missing_labels = kwargs
        return dict_missing_labels

    @property
    def trials(self):
        return self.n_trials

    def n_trials(self, n):
        self.n_trials = n

    @staticmethod
    def print_stats_time(execution_times, error_rate):
        output_str = 'mean_execution_time:{};std:{};error_rate:{}'.format(
            np.mean(execution_times), np.std(execution_times), error_rate)
        logger.info(str(output_str))
        print(output_str)

