import math
from pyspark.mllib.linalg import DenseVector

def class_mass_norm(vector, broadcast_labels, broad_l):
    priors = dict((key,val/broad_l.value ) for key,val in broadcast_labels.value.items() if not math.isnan(key))
    return DenseVector([val*priors[idx+1] for idx,val in enumerate(vector.tolist())])