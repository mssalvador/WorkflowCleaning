# the usual include statements
import os
import sys
import importlib
import argparse
import pyspark

package_dict = {
    'semisupervised.zip': './semisupervised', 'cleaning.zip': './cleaning',
    'classification.zip': './classification', 'shared.zip': './shared', 'examples.zip': './examples'}

for zip_file, path in package_dict.items():
    if os.path.exists(zip_file):
        sys.path.insert(0, zip_file)
    else:
        sys.path.insert(0, path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Runs a Pyspark job for workflow')
    parser.add_argument('--job', type=str, required=True, dest='job_name',
                        help='The name of the module that should be executed. '
                             '(ex. semi-supervised runs jobs in semi-supervised package')
    parser.add_argument('--job_args', nargs='*', help='The settings for the particular workflow '
                                                      '(ex. Algorithm=Kmeans, Standardize=True, k=20')
    parser.add_argument('--input_data', type=str, help='The location of the input data file.'
                                                       ' (ex. /home/user/data.txt)')
    parser.add_argument('--features', type=str, nargs='*', help='The feature columns for the dataset.'
                                                                ' (ex. a b c d ... x y)')
    parser.add_argument('--id', type=str, nargs='*', help='The identification column for the dataset')
    parser.add_argument('--labels', type=str, help='Labels on the training dataset.')

    args = parser.parse_args()
    print('Called with arguments: {}'.format(str(args)))

    all_args = dict()
    if args.job_args:
        all_args['algo_params'] = dict(arg.split('=') for arg in args.job_args)
    all_args['features'] = args.features
    all_args['input_data'] = args.input_data
    all_args['id'] = args.id
    all_args['labels'] = args.labels

    sc = pyspark.SparkContext(master='local[*]', appName=args.job_name)

    job_module = importlib.import_module('{:s}'.format(args.job_name))
    try:
        data_frame = job_module.run(sc, **all_args)
        data_frame.show()
    except TypeError as te:
        print('Did not run',te) # make this more logable...



    # from pyspark import SparkContext
    # from pyspark.sql import SparkSession
    # from pyspark.sql import functions as F
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    #
    # helix = PARQUET_PATH+'double_helix.parquet'
    #
    # sc = SparkContext.getOrCreate()
    # spark_session = SparkSession(sc)
    # spark_session.conf.set("spark.sql.crossJoin.enabled", "true")
    #
    # id_col = np.array([0, 1, 2, 3])
    # # np.random.shuffle(id_col)
    # data = {
    #     'label': [0.0, 1.0] + 2 * [None],
    #     'x': np.array([0., 0.9, 0.1, 0.85]),
    #     'y': np.array([0., 0.9, 0.1, 0.85]),
    #     'z': np.array([0., 0.9, 0.1, 0.85]),
    # }
    # pdf = pd.DataFrame(data, columns=['label', 'x', 'y', 'z'])
    # pdf['id'] = id_col
    #
    # #df_input = spark_session.createDataFrame(pdf)
    # df_input = spark_session.read.parquet(helix)
    # summed_transition = lp.label_propagation(
    #     sc= sc, data_frame= df_input,
    #     label_col= 'label', id_col= 'id',
    #     feature_cols= ['x','y','z'], k= 2, tol=0.01,
    #     max_iters=20, sigma=0.43, eval_type='cmn', priors= [0.3, 0.7])
    #
    # #print(summed_transition.filter(F.col('row_label') != float('nan')).take(2))
    # summed_transition.show(truncate=False)
    # combined_hack_df = df_input.select(['x','y','z','id']).alias('a').join(
    #     summed_transition.alias('b'), on= F.col('a.id') == F.col('b.row'),
    #     how= 'inner').drop('b.row')
    # combined_hack_df.drop('row_trans').show()
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # pandas_transition = combined_hack_df.toPandas()
    # for i in range(2):
    #     ax.scatter(
    #         pandas_transition[pandas_transition['label'] == i]['x'],
    #         pandas_transition[pandas_transition['label'] == i]['y'],
    #         pandas_transition[pandas_transition['label'] == i]['z'])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.set_title('Our test dataset a double helix')
    # plt.show()

