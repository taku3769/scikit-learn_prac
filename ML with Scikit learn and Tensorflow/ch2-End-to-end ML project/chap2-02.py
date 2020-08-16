'''
Practice code for "O'Reilly Hands-On Machne Learning with Scikit-Learn & TensorFlow"

Ver. 1.8
Includes:
- Fetch data
- Load data
- Stratification
- Ordinal encoder
- One-hot encodin
- Simple imputer
- Attribute adder
- Transformation pipeline

'''


import os
import sys
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL=DOWNLOAD_ROOT+"data/sets/housing/housing.tgz"
HOUSING_PATH=r"C:\Users\Taku\Google ドライブ\Codes\Machine Learning with Scikit learn and Tensorflow\data_files"

'''**************************************************************************'''
def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    ''' Get housing data from the URL, save, and extract ''' 
    if not os.path.isdir(housing_path):
        print('fetch: housing_path=',housing_path) 
        print('fetch: housing_url=',housing_url) 
        os.makedirs(housing_path)
        tgz_path=os.path.join(housing_path,"housing.tgz")
        urllib.request.urlretrieve(housing_url,tgz_path)
        housing_tgz=tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()

'''**************************************************************************'''
def load_housing_data(housing_path=HOUSING_PATH):
    ''' Load data file in csv using Panda ''' 
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)
    #return pd.read_csv(csv_path,encoding="cp932")

'''**************************************************************************'''
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[ train_indices], data.iloc[ test_indices]

'''**************************************************************************'''
def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio*2**32

def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[ id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set],data.loc[in_test_set]

'''**************************************************************************'''
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    # Custom transformer
    def __init__(self,add_bedrooms_per_room=True):  #no *args or **kargs
        self.add_bedrooms_per_room=add_bedrooms_per_room

    def fit(self,X,y=None):
        return self #nothing else to do

    def transform(self,X,y=None):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
        rooms_per_household=X[:,rooms_ix]/X[:,households_ix]
        population_per_household=X[:,population_ix]/X[:,households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

'''**************************************************************************'''
def main():
    np.set_printoptions(threshold=10) # Ndarray display threshold to avoid hiding some columns
    print('HOUSING_PATH=',HOUSING_PATH) 
    print('HOUSING_URL=',HOUSING_URL) 
    fetch_housing_data(HOUSING_URL,HOUSING_PATH)
    print('After fetch_housing_data') 
    housing = load_housing_data(HOUSING_PATH)
    print('After load_housing_data') 
    print(housing.head())

    # INFO statement
    print("\nINFO statement:")
    print(housing.info())

    # Value counts
    print("\nValue counts:")
    print(housing["ocean_proximity"].value_counts())

    # "describe" statement for summary
    print("\nDESCRIBE statement:")
    print(housing.describe())

    # Plot data
    #housing.hist(bins=50,figsize =(20,15))
    #plt.show()

    # Test set sampling - random vs stratification  
    housing["income_cat"]=pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])
    housing["income_cat"].hist()
    #plt.show()

    # Random test set
    rand_train_set, rand_test_set = train_test_split( housing, test_size = 0.2, random_state = 42)

    # Stratification of data
    print("\nStratify housing data:")
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    print(split.split(housing,housing["income_cat"]))
    print(len(list(split.split(housing,housing["income_cat"]))))

    ic = 0
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        ic += 1
        print("ic = ",ic)
        print(len(train_index),train_index)
        print(len(test_index),test_index)
        #sys.exit()
        strat_train_set=housing.loc[train_index]
        strat_test_set=housing.loc[test_index]
        strat_full_set=housing

    #rts = (rand_test_set["income_cat"].value_counts()/len(rand_test_set)).sort_index()
    #sts = (strat_test_set["income_cat"].value_counts()/len(strat_test_set)).sort_index()
    #sfs = (strat_full_set["income_cat"].value_counts()/len(strat_full_set)).sort_index()
    #print('rand_test:  \n{0}'.format(rts))
    #print('strat_test: \n{0}'.format(sts))
    #print('strat_full: \n{0}'.format(sfs))

    # Revert training set
    print("\nRevert training set:")
    housing = strat_train_set.drop("median_house_value",axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_cat = housing[["ocean_proximity"]]
    housing_cat_head = housing_cat.head(10)

    print("housing_cat.head(10) = {}".format(housing_cat.head(10)))
    #print("housing_cat.head(10) = {}".format(housing_cat_head))

    print("\nOrdinal encoder:")
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print("housing_cat_encoded = {0}".format(housing_cat_encoded[:10]))
    print("ordinal_encoder.categories_ = {0}".format(ordinal_encoder.categories_))

    print("\nOne-hot encoder:")
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    print("housing_cat_1hot = {0}".format(housing_cat_1hot))
    print("housing_cat_1hot.toarray() = {0}".format(housing_cat_1hot.toarray()))

    # Simple imputer
    imputer = SimpleImputer(strategy="median")
    housing_num_only = housing.drop("ocean_proximity",axis=1)
    imputer.fit(housing_num_only)
    print("imputer.statistics_ = {0}".format(imputer.statistics_))
    print("housing_num_only.median() = {0}".format(housing_num_only.median()))

    # Attribute adder
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    # Transformation pipeline
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy ="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
        ])

    print("num_pipeline = {0}".format(type(num_pipeline)))
    housing_num_tr = num_pipeline.fit_transform(housing_num_only)
    print("housing_num_tr = {0}".format(housing_num_tr))

    num_attribs = list(housing_num_only)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",OneHotEncoder(),cat_attribs),])

    housing_prepared = full_pipeline.fit_transform(housing)

    return


'''**************************************************************************'''

if __name__ == "__main__":
    main()
