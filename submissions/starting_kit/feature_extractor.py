import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass
    def transform(self, X_df):
        X_encoded = X_df

        path = os.path.dirname(__file__)
        award = pd.read_csv(os.path.join(path, 'award_notices_RAMP.csv.zip'), compression='zip',
                            low_memory=False)
        # obtain features from award
        award['Name_processed'] = award['incumbent_name'].str.lower()
        award['Name_processed'] = award['Name_processed'].str.replace('[^\w]','')
        award_features = award.groupby(['Name_processed'])['amount'].agg(['count','sum'])
        award_bids = award.groupby(['Name_processed'])['number_of_received_bids'].agg(['mean','sum'])
        #award_features = award.groupby(['Name_processed','end_year'])['amount','number_of_received_bids'].agg(['count','sum'])
        #award_features['mean_win'] = award_features['number_of_received_bids']['sum'] / (af['number_of_received_bids']['count'] + 1)
        
       
        
        def process_City(X):
            good_citys = ['MULHOUSE CEDEX 2', 'VAUCRESSON CEDEX', 'BOISSY L\'AILLERIE', 'YATE',
               'VENDARGUES CEDEX', 'MARCY L\'ETOILE', 'CHAUMONT', 'ST LERY', 'SOCHAUX',
               'ROCQUANCOURT', 'HENNEBONT CEDEX', 'MONTEREAU CEDEX', 'BAR SUR LOUP',
               'MAGNY LES HAMEAUX CEDEX', 'MARE', 'MARDYCK', 'MARIGNANE CEDEX',
               'ANGOULEME CEDEX 9', 'LE MESNIL ST DENIS CEDEX', 'KOUMAC',
               'MARNE LA VALLEE CEDEX 2', 'ST NAZAIRE CEDEX', 'BRIANCON', 'KONE',
               'ST OUEN CEDEX', 'COLOMBES CEDEX', 'ILLKIRCH CEDEX', 'OMEY',
               'AMBOISE CEDEX', 'VOH', 'STRASBOURG CEDEX 1', 'AMBOHIDRATRIMO',
               '92930 PARIS LA DEFENSE CED', 'THIO', 'NOUMEA CEDEX', 'BOULOUPARIS',
               'NEPOUI', 'CORBEIL ESSONNES CEDEX', 'BOURAIL', 'NANTERRE CEDEX',
               'EVRY CEDEX', 'SABLE SUR SARTHE CEDEX', 'MARLY LE ROI CEDEX',
               'HELLEMMES LILLE', 'MEUDON CEDEX', 'ETAULE', 'DRANCY CEDEX',
               'MONTPOUILLAN', 'VILLENEUVE D ASCQ CEDEX', 'ST JULIEN DE CHEDON']
            return X['City'].apply(lambda x : good_citys.index(x) if x in good_citys else len(good_citys)).values[:, np.newaxis]
        city_transformer = FunctionTransformer(process_City, validate=False)

        def zipcodes(X):
            zipcode_nums = pd.to_numeric(X['Zipcode'], errors='coerce')
            return zipcode_nums.values[:, np.newaxis]
        zipcode_transformer = FunctionTransformer(zipcodes, validate=False)

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])

        def process_date(X):
            date = pd.to_datetime(X['Fiscal_year_end_date'], format='%Y-%m-%d')
            return np.c_[date.dt.year, date.dt.month, date.dt.day]
        date_transformer = FunctionTransformer(process_date, validate=False)
        
        def process_APE(X):
            APE = X['Activity_code (APE)'].str[:3]
            return pd.to_numeric(APE).values[:, np.newaxis]
        APE_transformer = FunctionTransformer(process_APE, validate=False)
#         def merge_naive(X):
#             X['Name'] = X['Name'].str.lower()     
#             X['Name'] = X['Name'].str.replace('[^\w]','')
#             df = pd.merge(X, award_features, left_on='Name', 
#                           right_on='Name_processed', how='left')
#             return df[['count','sum']]
        def merge_naive(X):
            X['Name'] = X['Name'].str.lower()     
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, award_features, left_on=['Name'], right_on=['Name_processed'], how='left')
            return df[['count','sum']]
        merge_transformer = FunctionTransformer(merge_naive, validate=False)
        
        def merge_2(X):
            X['Name'] = X['Name'].str.lower()     
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, award_bids, left_on=['Name'], right_on=['Name_processed'], how='left')
            return df[['mean','sum']]
        merge2_transformer = FunctionTransformer(merge_2, validate=False)
        
        def cpv_transform(X):
            def decompose(s):
                l = s.split()
                l = list(map(lambda x:x[:2], l))
                return ' '.join(l)
            cpvdf = award[['Name_processed','CPV_classes']].dropna()
            cpvdf_group = cpvdf.groupby('Name_processed')[['CPV_classes']].count()
            cpvdf_group['classes'] = cpvdf.groupby('Name_processed')['CPV_classes'].apply(lambda x : ' '.join(x))
            cpvdf_group['classes'] = cpvdf_group['classes'].apply(decompose)
            vectorizer = TfidfVectorizer(decode_error = 'replace')
            classvalue = vectorizer.fit_transform(cpvdf_group['classes'])
            newdf = pd.DataFrame(classvalue.toarray(), index = cpvdf_group.index)
            columns = newdf.columns
            X['Name'] = X['Name'].str.lower()     
            X['Name'] = X['Name'].str.replace('[^\w]','')
            df = pd.merge(X, newdf, left_on=['Name'], right_on=['Name_processed'], how='left')
            return df[columns]
        cpv_transformer = FunctionTransformer(cpv_transform, validate=False)
        
        def macron(X):
            macron_binary = X['Year'].apply(lambda x : 1 if x >= 2017 else 0)
            return macron_binary.values[:, np.newaxis]
        Macron_transformer = FunctionTransformer(macron, validate=False)
        
        
        #merge_transformer = FunctionTransformer(merge_naive, validate=False)

        num_cols = ['Legal_ID', 'Headcount', 
                    'Fiscal_year_duration_in_months', 'Year']
        zipcode_col = ['Zipcode']
        date_cols = ['Fiscal_year_end_date']
        APE_col = ['Activity_code (APE)']
        merge_col = ['Name']
        drop_cols = ['Address']

        preprocessor = ColumnTransformer(
            transformers=[
                ('zipcode', make_pipeline(zipcode_transformer, SimpleImputer(strategy='median')), zipcode_col),
                ('num', numeric_transformer, num_cols),
                ('date', make_pipeline(date_transformer, SimpleImputer(strategy='median')), date_cols),
                ('APE', make_pipeline(APE_transformer, SimpleImputer(strategy='median')), APE_col),
                ('macron', make_pipeline(Macron_transformer, SimpleImputer(strategy='median')), ['Year']),
                ('merge', make_pipeline(merge_transformer, SimpleImputer(strategy='median')), merge_col),
                ('merge_bids', make_pipeline(merge2_transformer, SimpleImputer(strategy='median')), merge_col),
                #('merge_cpv', make_pipeline(cpv_transformer, SimpleImputer(strategy='median')), merge_col),
                ('cate', city_transformer, ['City']),
                ('drop cols', 'drop', drop_cols),
                ])

        X_array = preprocessor.fit_transform(X_encoded)
        return X_array