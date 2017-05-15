import sys
import os
from ipywidgets import widgets
from IPython.display import display, Javascript, HTML

from pyspark.sql import SQLContext, Window
from pyspark.sql import functions as F

class DataIO:
    'This class contains the data-import class for Cleaning'

    def __init__(self, sc, feature_path, company_path):
        # some small assertations that cover if the files are there
        assert os.path.exists(path=feature_path), "features data frame does not exist! Consider changing directory"
        assert os.path.exists(path=company_path), "company data frame does not exist! Consider changing directory"

        self.feature_path = feature_path
        self.company_path = company_path
        self.sc = sc
        self.sqlContext = SQLContext(sc)
        self._import_features_df = self.import_features_df()
        self._import_companies_df = self.import_companies_df()

    def import_features_df(self):
        return self.sqlContext.read.parquet(self.feature_path)


    def import_companies_df(self):
        return self.sqlContext.read.parquet(self.company_path)

    def show_features(self):
        self._import_companies_df.show()
        self._import_features_df.show()

    def get_latest_company(self, partion_col, order_col):
        '''
        
        :param companies_df: 
        :param partion_col: 
        :param order_col: 
        :return: altered imported_comp_df 
        '''

        assert isinstance(partion_col, list), "partion_col is not a list."
        assert isinstance(order_col, list), "partion_col is not a list."

        rank_by_cols = [F.col(i) for i in partion_col]
        order_by_cols = [F.col(i) for i in order_col]

        window_rank = Window.partitionBy(*rank_by_cols).orderBy(*order_by_cols)

        return (self._import_companies_df
                .withColumn("rank", F.rank().over(window_rank))
                .filter((F.col("rank") == 1) & (F.col("sekvensnr") == 0))
                .select("cvrNummer", "vaerdi")
                .withColumnRenamed(existing="vaerdi", new="navn")
                .orderBy("cvrNummer")
         )

    def mergeCompanyFeatureData(self, output_cols = []):

        clean_company_df = self.get_latest_company(["cvrNummer"],["periode_gyldigFra"]) #Get company on the right form

        if not output_cols:
            output_cols = self._import_features_df.columns+[i for i in clean_company_df.columns if i not in self._import_features_df.columns]

        joined_df = (self._import_features_df
                     .filter((F.col("kortBeskrivelse") == "APS") | (F.col("kortBeskrivelse") == "AS"))
                     .join(clean_company_df,("cvrNummer"),"inner")
                     )

        value_types = ("int", "float", "double", "long")
        value_columns = [i[0] for i in joined_df.dtypes if i[1] in value_types]
        #print(value_columns) # show columns that contain values

        return (joined_df
                .distinct()
                .na
                .fill(0.0, value_columns)
                )
