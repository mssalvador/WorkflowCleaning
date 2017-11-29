# TODO:
# Generisk DataIO med Widget interface
# omsæt .parquet/json/text (osv. datatyper) til dataframes
# skal kunne bruges i alle WFs


import logging

logger_data_import = logging.getLogger(__name__)
logger_data_import.setLevel(logging.DEBUG)
logger_file_handler_parameter = logging.FileHandler('/tmp/workflow_classification.log')
logger_formatter_parameter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger_data_import.addHandler(logger_file_handler_parameter)
logger_file_handler_parameter.setFormatter(logger_formatter_parameter)

from pyspark import SparkContext, SQLContext
from pyspark.sql import types as T
from IPython import display
from ipywidgets import widgets
from functools import partial

sc = SparkContext.getOrCreate()
sql_context = SQLContext.getOrCreate(sc)


class GeneralDataImport(object):
    """
    Data object to handle importation of various types of data
    """

    counter = 0
    file_ending = {'txt': 'text', 'csv': 'csv', 'parquet': 'parquet', 'jbdc': 'jbdc', 'json': 'json'}

    def __init__(self, path=None, **kwargs):
        """
        Constructor for GeneralDataImport
        :param path: path to data
        :param kwargs: if one has prior knowledge then it can be set into kwargs
        """

        # Can be set initially, but widget is runs no matter what.
        self._path_to_data = path
        self._standardize = kwargs.pop('standardize', False)
        self._all_columns = kwargs.pop('cols', [])
        self._list_structfield_id = None
        self._list_structfield_label = None
        self._list_structfield_features = None
        self._data_frame = None

        # Run the selection method
        self.select_file(**kwargs)

        logger_data_import.info("GeneralDataImport object created. {}".format(str(GeneralDataImport)))

    @property
    def data_frame(self):
        """
        Property for data_frame such that one can use object.data_frame instead of calling a function
        :return: Pyspark.dataframe
        """

        # Import pyspark functions
        import pyspark.sql.functions as F

        # Checks first if there is a data frame, outputs None.
        if self._data_frame is None:
            logger_data_import.warning("Data frame has not been set")
            return

        # Create casted features and labels
        feature_types = [GeneralDataImport.cast_to_right_type(feature) for feature in self._list_structfield_features]
        id_types = [F.col(id.name).cast('string').alias(id.name) for id in self._list_structfield_id]
        label_types = list(map(lambda c: F.col(c.name).cast('double').alias(c.name), self._list_structfield_label))

        # Log the selected features and labels
        logger_data_import.info(
            "Data frame exported with columns: {}, missing: {}"
                .format(self._list_structfield_label + self._list_structfield_id + self._list_structfield_features,
                        set(self._all_columns) - set(self._list_structfield_label + self._list_structfield_id + self._list_structfield_features)
                        )
        )

        return self._data_frame.select(id_types+label_types+feature_types)

    @property
    def all_columns(self):
        return list(map(lambda x: x.name, self._all_columns))

    @property
    def list_label(self):
        return list(map(lambda x: x.name, self._list_structfield_label))

    @property
    def list_id(self):
        return list(map(lambda x: x.name, self._list_structfield_id))

    @property
    def list_features(self):
        return list(map(lambda x: x.name, self._list_structfield_features))

    @property
    def standardize(self):
        return self._standardize

    @list_features.setter
    def list_features(self, list_feature):
        self._list_structfield_features = [T.StructField(feature, T.FloatType(), True)
                                           for feature in list_feature]

    @list_label.setter
    def list_label(self, list_label):
        self._list_structfield_label = [T.StructField(label, T.StringType(), True)
                                        for label in list_label]

    @list_id.setter
    def list_id(self, list_id):
        self._list_structfield_id = [T.StructField(ids, T.IntegerType(), True)
                                     for ids in list_id]

    def __del__(self):
        GeneralDataImport.counter = 0
        self._data_frame = None
        self._all_columns = None
        self._list_structfield_features = None
        self._list_structfield_id = None
        self._list_structfield_label = None
        self._path_to_data = ''

    def __repr__(self):
        return "GeneralDataImport( {:s} )".format(self._path_to_data)

    def __str__(self):
        return "Data from: {}".format(self._path_to_data)

    def select_file(self, **kwargs):
        """
        Write the path to file, should be an interactive kind of method...
        :param self:
        :return:
        """
        # import statements
        from IPython import display
        from ipywidgets import widgets

        # Widgets to be used
        text_import_file = widgets.Text(value= self._path_to_data, description= "path")
        button_import_file = widgets.Button(description= "Import file!")
        checkbox_standardize_data = widgets.Checkbox(value= False, description= "Standardize Data:")

        # Inline method for button handler
        def button_import_on_click(b):
            logger_data_import.info("File selected: {:s} - standardize is : {}"
                                    .format(text_import_file.value, checkbox_standardize_data.value))
            self._path_to_data = text_import_file.value
            self._standardize = checkbox_standardize_data.value
            # print(b.description)
            GeneralDataImport.counter += 1
            self._data_frame = GeneralDataImport.import_data(self._path_to_data, **kwargs)

            # set up for all columns and cleanup for labels and features
            self._all_columns = self._data_frame.schema
            self._list_structfield_id = [ids for ids in self._all_columns if ids.name in kwargs.get('idCols', [])]
            self._list_structfield_features = [feature for feature in self._all_columns
                                               if feature.name in kwargs.get('featureCols', [])]
            self._list_structfield_label = [lab for lab in self._all_columns
                                            if lab.name in kwargs.get('labelCols', [])]

        # register button event and show widgets
        button_import_file.on_click(button_import_on_click)
        display.display(widgets.HBox([text_import_file, button_import_file, checkbox_standardize_data]))

    @staticmethod
    def cast_to_right_type(name_and_type):
        """
        Method to check if the datatype is not string
        :param name_and_type: Pyspark StructType
        :return: Pyspark column that is either casted to a FloatType or is unchanged.
        """

        # Imports
        import pyspark.sql.functions as F
        if name_and_type.dataType == T.StringType():
            return F.col(name_and_type.name).cast('float').alias(name_and_type.name)
        else:
            return F.col(name_and_type.name)

    @staticmethod
    def extract_type(path_string):
        """
        Filters out the file type .csv, .parquet, etc.
        :param path_string: string with file path
        :return: ".csv", ".parquet, or other file type
        """
        import re
        return str(re.search('\.\w+$', path_string, re.IGNORECASE).group())

    @staticmethod
    def import_data(path, **kwargs):
        """
        Loads the data from a file and saves it to a Pyspark.dataframe
        :param path: path to data
        :param kwargs: key-value pair arguments for options
        :return: Pyspark.dataframe
        """

        # Checks if no kwarg have been submitted.
        if kwargs.__len__() == 0:
            kwargs = {'header': True, 'encoding': 'utf-8'}

        data_format = GeneralDataImport.extract_type(path)

        # Special case for json. There might be a schema hidding somewhere.
        if data_format[1:] != 'json':
            return (sql_context
                    .read
                    .format(GeneralDataImport.file_ending[data_format[1:]])
                    .options(**kwargs)
                    .load(path))
        else:
            return (sql_context
                    .read
                    .json(path))

    def select_id(self):
        set_all_columns = set([
            col.name for col in self._all_columns])
        set_features = set([
            col.name for col in self._list_structfield_features])
        set_labels = set([
            col.name for col in self._list_structfield_label])
        set_ids = set([
            col.name for col in self._list_structfield_id])
        list_avail_columns = list(
            set_all_columns - set_features - set_labels - set_ids)

        widget_select_id = widgets.SelectMultiple(
            value=[], options=list_avail_columns, description="Select Features"
        )

        def obs_id(change):
            if change.new != change.old:
                self._list_structfield_id = [i for i in self._all_columns if
                                             i.name in widget_select_id.value]

        widget_select_id.observe(obs_id, names='value')
        display.display(widget_select_id)

    def select_features(self):
        set_all_columns = set([
            col.name for col in self._all_columns])
        set_features = set([
            col.name for col in self._list_structfield_features])
        set_labels = set([
            col.name for col in self._list_structfield_label])
        set_ids = set([
            col.name for col in self._list_structfield_id])
        list_avail_columns = list(
            set_all_columns - set_features - set_labels - set_ids)

        widget_select_feature = widgets.SelectMultiple(
            value= [], options= list_avail_columns, description= "Select Features"
        )

        def obs_feature(change):
            if change.new != change.old:
                self._list_structfield_features = [i for i in self._all_columns if
                                             i.name in widget_select_feature.value]

        widget_select_feature.observe(obs_feature, names= 'value')
        display.display(widget_select_feature)

    def select_labels(self):
        set_all_columns = set([
            col.name for col in self._all_columns])
        set_features = set([
            col.name for col in self._list_structfield_features])
        set_labels = set([
            col.name for col in self._list_structfield_label])
        set_ids = set([
            col.name for col in self._list_structfield_id])
        list_avail_columns = list(
            set_all_columns - set_features - set_labels - set_ids)

        widget_select_labels = widgets.SelectMultiple(
            value= [], options= list_avail_columns, description= "Select Features"
        )

        def obs_feature(change):
            if change.new != change.old:
                self._list_structfield_label = [i for i in self._all_columns if
                                             i.name in widget_select_labels.value]
        widget_select_labels.observe(obs_feature, names= 'value')
        display.display(widget_select_labels)