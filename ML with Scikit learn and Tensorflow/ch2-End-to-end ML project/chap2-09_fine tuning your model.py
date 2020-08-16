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
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV 


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
rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    # Custom transformer
    def __init__(self,add_bedrooms_per_room=True):  #no *args or **kargs
        self.add_bedrooms_per_room=add_bedrooms_per_room

    def fit(self,X,y=None):
        return self #nothing else to do

    def transform(self,X,y=None):
        #rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
        rooms_per_household=X[:,rooms_ix]/X[:,households_ix]
        population_per_household=X[:,population_ix]/X[:,households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]

def display_scores(scores):
    print("    Scores:",scores)
    print("    Mean:",scores.mean())
    print("    Standard deviation:",scores.std()) 

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
    # Not necessarily the best sampling method. Ex. Sex can influence the median income -> right fraction of male/female is critical 
    rand_train_set, rand_test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

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
        #strat_full_set=housing

    #rts = (rand_test_set["income_cat"].value_counts()/len(rand_test_set)).sort_index()
    #sts = (strat_test_set["income_cat"].value_counts()/len(strat_test_set)).sort_index()
    #sfs = (strat_full_set["income_cat"].value_counts()/len(strat_full_set)).sort_index()
    #print('rand_test:  \n{0}'.format(rts))
    #print('strat_test: \n{0}'.format(sts))
    #print('strat_full: \n{0}'.format(sfs))

    # Separate predictors and labels 
    print("\nSeparate predictors and labels:")
    housing = strat_train_set.drop("median_house_value",axis=1)   # <- Predictor data
    housing_labels = strat_train_set["median_house_value"].copy() # <- Labels
    housing_cat = housing[["ocean_proximity"]]                    # Non-numeric categories
    print("housing_cat.head(10) = {}".format(housing_cat.head(10)))

    ''' Sklearn - Simple imputer '''
    imputer = SimpleImputer(strategy="median")
    housing_num_only = housing.drop("ocean_proximity",axis=1)
    imputer.fit(housing_num_only)
    print("imputer.statistics_ = {0}".format(imputer.statistics_))
    print("housing_num_only.median() = {0}".format(housing_num_only.median()))
    X = imputer.transform(housing_num_only)
    housing_tr = pd.DataFrame(X,columns=housing_num_only.columns)
    print('housing_tr.info() : ')
    print(housing_tr.info())

    ''' Encording '''
    print("housing_cat_encoded = {0}".format(housing_cat[:10]))
    # Ordinal encoder : replace categorical attributes into numbers
    # Issue with this method is the "distance" between the numerical values
    ##print("\nOrdinal encoder:")
    ##ordinal_encoder = OrdinalEncoder()
    ##housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    ##print("housing_cat_encoded = {0}".format(housing_cat_encoded[:10]))
    ##print("ordinal_encoder.categories_ = {0}".format(ordinal_encoder.categories_))

    # One-hot encorder: Split categories and label only 0 or 1 
    # This way can avoid "distance" problem of the ordinal encorder
    # Output is a SiPy sparse matrix. User toarray() to convert to numpy array 
    print("\nOne-hot encoder:")
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    #print("housing_cat_1hot = {0}".format(housing_cat_1hot))
    #print("housing_cat_1hot.toarray() = {0}".format(housing_cat_1hot.toarray()))

    # Attribute adder
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    housing_extra_attribs = attr_adder.transform(housing.values)

    # Transformation pipeline
    num_pipeline = Pipeline([
        ('imputer',SimpleImputer(strategy ="median")),
        ('attribs_adder',CombinedAttributesAdder()),
        ('std_scaler',StandardScaler()),
        ])

    #print("num_pipeline = {0}".format(type(num_pipeline)))
    housing_num_tr = num_pipeline.fit_transform(housing_num_only)
    #print("housing_num_tr = {0}".format(housing_num_tr))

    num_attribs = list(housing_num_only)
    cat_attribs = ["ocean_proximity"]
    full_pipeline = ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",OneHotEncoder(),cat_attribs),])

    print("housing.head(5): ",housing.head(5))
    housing_prepared = full_pipeline.fit_transform(housing)
    print("housing_parepared: ",pd.DataFrame(housing_prepared).iloc[:5])

    ''' Training and evaluating on the training set '''
    # Perform linear regression 
    print('Linear regression:')
    lin_reg = LinearRegression() 
    lin_reg.fit(housing_prepared, housing_labels)

    #print("housing.head(5): ",housing.head(5))
    ''' 
    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    print("len(some_data): ",len(some_data))
    print("some_data: ",some_data)
    print("len(some_labels): ",len(some_labels))
    print("some_labels: ",some_labels)
    some_data_prepared = full_pipeline.transform(some_data)  # Output is a numpy array
    #print("some_data_prepared: ",pd.DataFrame(some_data_prepared))
    print("Labels: ",list(some_labels))
    ''' 
    print("Labels: ",housing_labels)
    #print("Labels: ",list(housing_labels))

    #np.set_printoptions(threshold=np.inf)

    # Compute RMSE
    #some_predictions = lin_reg.predict(some_data_prepared)
    #print("Predictions:", type(some_predictions))
    #print("len(Predictions):", len(some_predictions))
    #print("Predictions:", some_predictions)
    #lin_mse = mean_squared_error(some_labels,some_predictions)
    housing_predictions = lin_reg.predict(housing_prepared)
    print("len(housing_predictions):", len(housing_predictions))
    print("Predictions:", housing_predictions)
    lin_mse = mean_squared_error(housing_labels,housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print('lin_rmse = {0}\n'.format(lin_rmse))


    # Perform Decision tree regressor
    print('Decision tree regressor:')
    tree_reg = DecisionTreeRegressor() 
    tree_reg.fit( housing_prepared, housing_labels)

    # Compute RMSE
    housing_predictions = tree_reg.predict(housing_prepared)
    print("Predictions:", housing_predictions)
    tree_mse = mean_squared_error(housing_labels,housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print('tree_rmse = {0}\n'.format(tree_rmse))
    print('')

    ''' Cross-validation ''' 
    print('*** Cross-validation ***')
    print('  Decision tree:')
    scores = cross_val_score(tree_reg,housing_prepared,housing_labels,
    scoring="neg_mean_squared_error",cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    display_scores(tree_rmse_scores)
    print('')

    print('  Linear regression:')
    lin_scores = cross_val_score(lin_reg,housing_prepared,housing_labels,
    scoring="neg_mean_squared_error",cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)
    print('')


    print('  Random forest regressor:')
    forest_reg = RandomForestRegressor() 
    forest_reg.fit(housing_prepared,housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    print("Predictions:", housing_predictions)
    forest_mse = mean_squared_error(housing_labels,housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print('forest_rmse = {0}\n'.format(forest_rmse))
    scores = cross_val_score(forest_reg,housing_prepared,housing_labels,
    scoring="neg_mean_squared_error",cv=10)
    forest_rmse_scores = np.sqrt(-scores)
    display_scores(forest_rmse_scores)
    print('')

    ''' Fine-tuning ''' 
    param_grid = [{'n_estimators':[3,10,30,],'max_features':[2,4,6,8]},
    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},
    ]
    forest_reg = RandomForestRegressor()
    grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
    grid_search.fit(housing_prepared,housing_labels)

    print('grid_search.best_params_ = {}'.format(grid_search.best_params_))

    print('grid_search.best_estimator_',grid_search.best_estimator_)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
        print(np.sqrt(-mean_score),params)

    feature_importances = grid_search.best_estimator_. feature_importances_
    print('feature_importances: \n{0}'.format(feature_importances))

    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"] 
    cat_encoder = full_pipeline.named_transformers_["cat"] 
    cat_one_hot_attribs = list( cat_encoder.categories_[0]) 
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print('sorted(fi,attr):')
    print(sorted(zip(feature_importances,attributes), reverse=True))


    return


'''**************************************************************************'''

if __name__ == "__main__":
    main()
