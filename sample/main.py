# the usual include statements

from pyspark import SparkContext
from pyspark import SQLContext
from shared import create_dummy_data
from shared import Plot2DGraphs
from shared import ParseLogFiles
import getpass
import os
import re

user = getpass.getuser()

if user == "sidsel":
    PARQUET_PATH = "/home/" + user + "/workspace/sparkdata/parquet/"

elif user == "svanhmic":
    PARQUET_PATH = "/home/" + user + "/workspace/data/DABAI/sparkdata/parquet/"

if __name__ == '__main__':
    # n_dimension = 2
    # n_clusters = 10
    # samples = 10000
    #
    # for i in [10000000]:
    #     means = create_dummy_data.create_means(n_dimension, n_clusters, 10)  # [[0, 0, 0], [3, 3, 3], [-3, 3, -3], [5, -5, 5]]
    #     stds = create_dummy_data.create_stds(n_dimension, n_clusters)  # [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    #     n_samples = create_dummy_data.create_partition_samples(i, n_clusters)  # [1000, 10000, 4000, 50]
    #     print(n_samples)
    #     df = create_dummy_data.create_normal_cluster_data_spark(n_dimension, n_samples, means, stds)
    #     #df.show(100)
    #     df.write.parquet(PARQUET_PATH+'normal_cluster_n_'+str(i)+'.parquet', mode='overwrite')

    f = '/home/svanhmic/workspace/results/DABAI/performancetest'
    files = list(map(lambda x: f+'/'+str(x), os.listdir(f)))
    data = []
    with open(files[0],'r') as file:
        for line in file:
            data.append(line)

    pdf = ParseLogFiles.divide_string(data)
    pdf