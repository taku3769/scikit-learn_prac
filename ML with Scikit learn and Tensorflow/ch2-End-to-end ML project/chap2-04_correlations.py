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

    ''' Visualize geographical data '''
    # Default scatter plot
    #housing.plot(kind="scatter",x="longitude",y="latitude") 
    #plt.show()

    # Default scatter plot with alpha
    #housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1) 
    #plt.show()

    # Default scatter plot with alpha
    #housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,
    #    s=housing["population"]/100,label="population",
    #    figsize=(10,7),c="median_house_value",
    #    cmap=plt.get_cmap("jet"),colorbar=True,
    #) 
    #plt.legend()
    #plt.show()

    # Correlations
    corr_matrix = housing.corr()
    print('corr_matrix:')
    print(corr_matrix)

    # Check how each attribute correlates with median house value
    print('corr_matrix[median_house_value]:')
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # Pandas scatter matrix
    attributes = ["median_house_value","median_income","total_rooms","housing_median_age"]
    scatter_matrix(housing[attributes],figsize=(12,8))
    plt.show()

    return


'''**************************************************************************'''

if __name__ == "__main__":
    main()
