# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:46:56 2023

@author: Hasan Emre
"""
#%% Iqmport Library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm, skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

# XGBoost
import xgboost as xgb

# warning
import warnings
warnings.filterwarnings("ignore") 


#%% Data Set and Problem Description (Veri Seti ve Problem Tanitimi)

# Column names set  (Sutun adlari belirlenir)
columns_name = ["MPG","Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]

# Space characters are removed and data is loaded.  (Bosluk karakterleri cikartilarak verilerin yuklenmesi saglanir.)
data = pd.read_csv("auto-mpg.data", names=columns_name, na_values="?", comment="\t", sep= " ",skipinitialspace= True)

# The "MPG" column name in the dataset is changed to "target".  (# Veri kumesindeki "MPG" sutunu adi "target" olarak degistirildi)
data = data.rename(columns={"MPG":"target"})


print(data.head())
print("Data shape: ",data.shape)

# Gives general information about the data  (veriler hakkinda genel bilgi verir)
data.info()

# Basic statistical properties of the dataset are printed on the screen  (Veri kumesinin temel istatistiksel ozellikleri ekrana yazdirilir )
describe = data.describe()


#%% Missing Value
# Calculates and prints the number of missing values in the data frame  (Veri cercevesindeki eksik değerlerin sayisini hesaplar ve ekrana yazdirir)
print(data.isna().sum())

# Fills the missing values in the "Horsepower" column with the average value of the column  ("Horsepower" sutunundaki eksik degerleri sutunun ortalama degeri ile doldurur)
data["Horsepower"] = data["Horsepower"].fillna(data["Horsepower"].mean()) 
# Visualizes the distribution of data in the "horsepower" column ("Horsepower" sutunundaki verilerin dagilimini gorselleştirir)
sns.distplot(data.Horsepower)

print(data.isna().sum())

#%% EDA

# Calculates the correlation matrix between features in the data frame and visualizes the Correlation matrix
# (Veri cercevesindeki ozellikler arasindaki korelasyon matrisini hesaplar ve Korelasyon matrisini gorsellestirir)
corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot= True, fmt = ".2f")
plt.title("Correlation btw feat")
plt.show()

#  Filters features above a certain correlation threshold and visualizes the correlation matrix between the filtered features
# (Belirli bir korelasyon esiginin uzerindeki ozellikleri filtreler ve filtrelenen ozellikler arasindaki korelasyon matrisini gorsellestirir)
threshold = 0.75
filtre = np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr(), annot=True, fmt = ".2f")
plt.title("Correlation btw feat")
plt.show()


"""
    multicollinearity
"""
# Visualizes multicollinearity between features  (Ozellikler arasindaki coklu dogrusalligi gorsellestirir)
sns.pairplot(data, diag_kind="kde", markers="+")
plt.show()

"""
    Cylinders and origin can be categorical (feature engineering)
"""

# Visualizes the categorical status of some features (Bazi ozelliklerin kategorik olma durumunu gorsellestirir)
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_count())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_count())


# Visualizes the distribution of each feature using a box plot (Bir kutu grafigi kullanarak her bir özelligin dagitimini gorsellestirir)

for c in data.columns:
    plt.figure()
    sns.boxplot(x = c, data = data, orient="v")
    

"""
outlier:   Horsepower and acceleration
"""

# %% Outlier

# We remove the outliers from the "Horsepower" and "Acceleration" columns (Aykiri degerleri "Beygir gucu" ve "Hizlanma" sutunlarindan kaldiriyoruz)

thr = 2

# Horsepower outliers
horsepower_desc = describe["Horsepower"]

q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp

top_limit_hp = q3_hp + thr*IQR_hp
bottom_limit_hp = q1_hp - thr*IQR_hp

filter_hp_botttom = bottom_limit_hp < data["Horsepower"]
filter_hp_top = data["Horsepower"] < top_limit_hp
filter_hp = filter_hp_botttom & filter_hp_top

data = data[filter_hp] # remove Horsepower outliers



# Acceleration outliers
acceleration_desc = describe["Acceleration"]

q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc

top_limit_acc = q3_acc + thr*IQR_acc
bottom_limit_acc = q1_acc - thr*IQR_acc

filter_acc_botttom = bottom_limit_hp < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc
filter_acc = filter_acc_botttom & filter_acc_top

data = data[filter_acc] # remove Acceleration outliers


#%%  Features Engineering
# skewness

# target dependent variable
# We draw the appropriate curve by calculating the mean (mu) and standard deviation (sigma) values
#( Ortalama (mu) ve standart sapma(sigma) degerlerini hesaplayarak uygun egriyi cizdiriyoruz)
sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu: {}, sigma: {}". format(mu, sigma))

#  Probability Plot
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()


# The np.log1p() function takes the logarithm of the given value and converts it by adding 1
# (np.log1p() islevi, verilen degerin logaritmasini alir ve 1 ekleyerek donusturur)
data["target"] = np.log1p(data["target"])

plt.figure()
sns.distplot(data.target, fit = norm)

(mu, sigma) = norm.fit(data["target"])
print("mu: {}, sigma: {}". format(mu, sigma))

# Probability Plot
plt.figure()
stats.probplot(data["target"], plot = plt)
plt.show()


# Feature - independent variable
# We sort from largest to smallest, calculating the "skew" value for each column and add it as a column
# (Her sutun icin "egrilik" degerini hesaplayarak buyukten kucuge siralariz ve sutun olarak ekliyoruz) 
skewed_feats = data.apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame(skewed_feats, columns = "skewed")

"""
Box Cox Transformatin
"""


#%% one hot encoding
# We convert the columns to strings to categorize them. Then it transforms it into new columns consisting of 0s and 1s with dummy variable transformation.
# (Sutunlari kategorilendirmek icin stringe ceviriyoruz. Daha sonra dummy degisken donusumu ile 0 ve 1'lerden olusan yeni sutunlara donusturur.)
data["Cylinders"] = data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)

data = pd.get_dummies(data)


#%% train test split - standardization

x = data.drop(["target"], axis = 1)
y = data.target

test_size = 0.9
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=42)


# Standardization
# The StandardScaler object is used to convert data to mean 0 and standard deviation 1
# (# StandardScaler nesnesi, verileri ortalama 0 ve standart sapma 1'e donusturmek icin kullanilir)

scaler = StandardScaler()  # RobustScaler
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#%%  Regression Models

# Linear Regression
# After creating the Linear Regression model, we training (fit())
# (Linear Regression modelini oluşturduktan sonra egitiyoruz (fit()) )
lr = LinearRegression()
lr.fit(x_train, y_train)
print("LR Coef: ", lr.coef_)

# We estimate the test data and calculate the MSE
# (Test verilerini tahmin ediyoruz ve MSE degerini hesapliyoruz)
y_predicted_dummy = lr.predict(x_test)
mse = mean_squared_error(y_test, y_predicted_dummy)
print("Linear Regression MSE: ", mse)

#%% Ridge Regression (L2)

# By determining the alpha value of the Ridge regression model in the logarithmic range, it makes hyperparameter adjustments and selects 
# the most appropriate alpha value on the training data. It then prints the best estimator and coefficients and visualizes the scores against the alpha value.
# (Ridge regresyon modelinin logaritmik araliktaki alfa degerini belirleyerek hiperparametre ayarlamalari yapar ve egitim verileri uzerinde 
# en uygun alfa degerini secer. Daha sonra en iyi tahmin ediciyi ve katsayilari yazdirir ve puanlari alfa degerine gore gorsellestirir.)

ridge = Ridge(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv = n_folds, scoring="neg_mean_squared_error", refit = True)
clf.fit(x_train, y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ", clf.best_estimator_.coef_)
ridge = clf.best_estimator_
print("Ridge Best Estimator: ", ridge)

y_predict_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test, y_predict_dummy)
print("Ridge MSE: ", mse)
print("--------------------------------------------------")


plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Ridge")


#%%  Lasso Regression (L1)

# By determining the alpha value of the Lasso regression model in the logarithmic range, it makes hyperparameter adjustments and selects 
# the most appropriate alpha value on the training data. It then prints the best estimator and coefficients and visualizes the scores against the alpha value.
# (Lasso regresyon modelinin logaritmik araliktaki alfa degerini belirleyerek hiperparametre ayarlamalari yapar ve egitim verileri uzerinde 
# en uygun alfa degerini secer. Daha sonra en iyi tahmin ediciyi ve katsayilari yazdirir ve puanlari alfa degerine gore gorsellestirir.)

lasso = Lasso(random_state=42, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv = n_folds, scoring="neg_mean_squared_error", refit = True)
clf.fit(x_train, y_train)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

print("Lasso Coef: ", clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ", lasso)

y_predict_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test, y_predict_dummy)
print("Lasso MSE: ", mse)
print("--------------------------------------------------")


plt.figure()
plt.semilogx(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")


#%%  ElasticNet Regression

# By determining the alpha value of the ElasticNet regression model in the logarithmic range, it makes hyperparameter adjustments and selects 
# the most appropriate alpha value on the training data. It then prints the best estimator and coefficients and visualizes the scores against the alpha value.
# (ElasticNet regresyon modelinin logaritmik araliktaki alfa degerini belirleyerek hiperparametre ayarlamalari yapar ve egitim verileri uzerinde 
# en uygun alfa degerini secer. Daha sonra en iyi tahmin ediciyi ve katsayilari yazdirir ve puanlari alfa degerine gore gorsellestirir.)

parametersGrid = {"alpha": alphas,
                  "l1_ratio": np.arange(0.0, 1.0, 0.05)}

eNet = ElasticNet(random_state=42, max_iter=10000)

clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring="neg_mean_squared_error", refit=True)
clf.fit(x_train, y_train)

best_eNet = clf.best_estimator_
best_coefs = best_eNet.coef_

print("ElasticNet Best Estimator: ", best_eNet)
print("ElasticNet Coefficients: ", best_coefs)

y_predict_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test, y_predict_dummy)
print("ElasticNet MSE: ", mse)


#%% Regression Results

"""
    StandardScaler()
        Linear Regression MSE: 0.0012023301858505278
        Ridge MSE: 0.001143259929053612
        Lasso MSE: 0.001113490141530847
        ElasticNet MSE: 0.0010229421793035102
            
     RobustScaler()
         Linear Regression MSE: 0.0011936460617962003
         Ridge MSE: 0.0011185782450321804
         Lasso MSE: 0.0010877708074630973
         ElasticNet MSE: 0.0010684329933544217
"""

#%% XGBoost Regression

# It trains the XGBoost regression model with the best hyperparameters using GridSearchCV, estimates the test data, 
# calculates the mean square error (MSE) and prints the result to the screen.
# (GridSearchCV kullanarak XGBoost regresyon modelini en iyi hiperparametrelerle egitir, test verilerini tahmin eder, 
# ortalama kare hatasini (MSE) hesaplar ve sonucu ekrana yazdirir.)

parametersGrid = {"nthread":[4],
                  "objective":["reg:linear"],
                  "learning_rate":[.03, 0.05, .07],
                  "max_depth":[5, 6 , 7],
                  "min_child_weight": [4],
                  "silent": [1],
                  "subsample": [0.7],
                  "colsample_bytree": [0.7],
                  "n_estimators": [500, 1000]}



model_xgb = xgb.XGBRegressor()

clf = GridSearchCV(model_xgb, parametersGrid, cv = n_folds, scoring="neg_mean_squared_error", refit=True, n_jobs= 5, verbose=True)

clf.fit(x_train, y_train)
model_xgb = clf.best_estimator_

y_predict_dummy = clf.predict(x_test)
mse = mean_squared_error(y_test, y_predict_dummy)
print("XGBoost MSE: ", mse)

# XGBoost MSE:  0.0010397543214997543


#%% Averaging Models

class AveragingModels():
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models, to fit the data in
    def fit(self, x, y):
        self.models_ = [clone(model) for model in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(x, y)
            
        return self
    
    # Now we do the predictions for cloned models and average them
    def predict(self, x):
        predictions = np.column_stack([model.predict(x) for model in self.models_])
        return np.mean(predictions, axis=1)
    
# Modelleri çağırmadan önce fonksiyonları kullanın
averaged_models = AveragingModels(models=(model_xgb, lasso))

# Modelleri eğitin
averaged_models.fit(x_train, y_train)

# Tahmin yapın
y_predict_dummy = averaged_models.predict(x_test)

# Hata ölçümünü hesaplayın
mse = mean_squared_error(y_test, y_predict_dummy)
print("Averaged Model MSE: ", mse)


# Averaged Model MSE:  0.0009938790754707649