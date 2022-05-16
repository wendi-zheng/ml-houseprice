import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# display options
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
np.set_printoptions(linewidth = 1000)
seperation_line = '*' * 50

# read data from csv
data_train = pd.read_csv("data_source/train.csv", encoding = "utf-8")
y = data_train["SalePrice"]
x = data_train.drop(["SalePrice", "Id"], axis = 1)


# seperate features by type
numerical_features = list(x.select_dtypes(exclude ="object").columns)
categorical_features = list(x.select_dtypes(include ="object").columns)

# normalize y
y = np.log(y)

# define cross-validation
def CV_RMSE(model, x_in, y_in):
    kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
    MSE = -1 * cross_val_score(model, x_in, y_in, scoring = "neg_mean_squared_error", cv = kf)
    return np.sqrt(MSE)

# define model
rf = RandomForestRegressor(random_state = 0)
gbr = GradientBoostingRegressor(n_estimators = 1000, learning_rate = 0.05, max_depth = 4)
xgb = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, max_depth = 4, n_jobs = 4)

# define modeling & evaluation processes
def eval(header_msg, x_in, y_in):
    print(seperation_line)
    print(header_msg)
    print("x shape: ", x_in.shape)
    print("Random Forest: ")
    rf.fit(x_in, y_in)
    rf_results = CV_RMSE(rf, x_in, y_in)
    print("    CV results: ", rf_results)
    print("    RMSE: %.5f" % (rf_results.mean()))
    gbr.fit(x_in, y_in)
    print("Gradient Boosting: ")
    gbr_results = CV_RMSE(gbr, x_in, y_in)
    print("    CV results: ", gbr_results)
    print("    RMSE: %.5f" % (gbr_results.mean()))
    print("XGBoosting: ")
    xgb.fit(x_in, y_in)
    xgb_results = CV_RMSE(xgb, x_in, y_in)
    print("    CV results: ", xgb_results)
    print("    RMSE: %.5f" % (xgb_results.mean()))

#################### without proper feature engineering #######################
x0 = x.copy()

# encoding
x0 = x0[numerical_features]

# drop missing values
cols_with_missing = [col for col in x0.columns if x0[col].isnull().any()]
x0 = x0.drop(columns = cols_with_missing, axis = 1)

# model evaluation
eval("without proper feature engineering: ", x0, y)

#################### with feature engineering #################################
# remove outliers
data_train.drop(data_train[data_train["GrLivArea"] > 4000].index, inplace = True)
data_train.drop(data_train[data_train["GarageArea"] > 1200].index, inplace = True)
data_train.drop(data_train[data_train["TotalBsmtSF"] > 3000].index, inplace = True)
data_train.drop(data_train[data_train["1stFlrSF"] > 3000].index, inplace = True)
data_train.drop(data_train[data_train["LotFrontage"] > 200].index, inplace = True)
data_train.drop(data_train[data_train["LotArea"] > 100000].index, inplace = True)
y = data_train["SalePrice"]
x = data_train.drop(["SalePrice", "Id"], axis = 1)
y = np.log(y)
x1 = x.copy()

# fill missing values
numerical_missing = ["LotFrontage", "MasVnrArea"]
x1[numerical_missing] = x1[numerical_missing].fillna(0)

categorical_missing = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
                       "GarageCond", "GarageType","GarageFinish", "GarageQual",
                       "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtCond",
                       "BsmtQual", "MasVnrType", "Electrical"]
x1[categorical_missing] = x1[categorical_missing].fillna("NotIncluded")

x1["GarageYrBlt"] = x1["GarageYrBlt"].fillna(x1["YearBuilt"])

# create new features
x1["TotalLivArea"] = x1["TotalBsmtSF"] + x1["1stFlrSF"] + x1["2ndFlrSF"]
x1["TotalPorchSF"] = x1["OpenPorchSF"] + x1["EnclosedPorch"] + x1["3SsnPorch"] + x1["ScreenPorch"]
x1["TotalFullBath"] = x1["FullBath"] + x1["BsmtFullBath"]
x1["TotalHalfBath"] = x1["HalfBath"] + x1["BsmtHalfBath"]
x1["HouseAge"] = x1["YrSold"] - x1["YearBuilt"]
x1["Remod"] = np.where(x1["YearRemodAdd"] == x1["YearBuilt"], 0, 1)
x1["HasGarage"] = x1["GarageArea"].apply(lambda x: 1 if x > 0 else 0)
x1["HasBsmt"] = x1["TotalBsmtSF"].apply(lambda x: 1 if x > 0 else 0)
x1["HasPool"] = x1["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
x1["HasPorch"] = x1["TotalPorchSF"].apply(lambda x: 1 if x > 0 else 0)
x1["Has2ndFlr"] = x1["2ndFlrSF"].apply(lambda x: 1 if x > 0 else 0)

# drop features
x1.drop(["Street", "Utilities", "Condition2"], axis = 1)

# transform numerical variables to categorical
cols_to_change = ["MSSubClass", "YearBuilt", "YearRemodAdd", "GarageYrBlt",
                  "MoSold", "YrSold"]
for col in cols_to_change:
    x1[col] = x[col].apply(str)
    numerical_features.remove(col)
    categorical_features.append(col)

# encoding
x1 = pd.get_dummies(x1).reset_index(drop = True)

# model evaluation
eval("with proper feature engineering: ", x1, y)
