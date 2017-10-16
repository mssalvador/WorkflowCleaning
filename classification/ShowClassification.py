from pyspark.ml.tuning import CrossValidatorModel
import matplotlib.pyplot as plt

def create_confusion_matrix(data_frame, labelCol=None, predictionCol=None):

    assert [labelCol, predictionCol] in data_frame.columns, ' Label Column or Prediction Column not in data_frame'
    grouped_labels = data_frame.groupBy([labelCol, predictionCol]).count()
    return grouped_labels


def show_classification_attributes(cross_validated_model):

    learning_algo = cross_validated_model.bestModel.stages[-1]
    if learning_algo.hasSummary:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=(20, 14))
        summary = learning_algo.summary

        print('The area under the curve is {}'.format(summary.areaUnderROC))
        attributes = []
        titles = ['F-measure by Threshold', 'Precision by Recall', 'Precision by Threshold', 'ROC',
                  'Recall by Threshold']
        attributes.append(summary.fMeasureByThreshold.toPandas())
        attributes.append(summary.pr.toPandas())
        attributes.append(summary.precisionByThreshold.toPandas())
        attributes.append(summary.roc.toPandas())
        attributes.append(summary.recallByThreshold.toPandas())
        # iterations = summary.totalIterations

        jdx = 0
        for idx, data_frame in enumerate(attributes):
            if idx % 3 == 0 and idx != 0:
                jdx += 1
            ax = axes[jdx, idx % 3]
            ax.plot(data_frame.columns[0],
                    data_frame.columns[1],
                    data=data_frame,
                    )
            ax.legend()
            ax.set_xlabel(data_frame.columns[0])
            ax.set_ylabel(data_frame.columns[1])
            ax.set_title(titles[idx])
        plt.show()
    else:
        print('No summary is avaible, consider your algorithm')
        return