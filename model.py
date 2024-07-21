import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio 
# import pandas_profiling as pp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.utils import resample 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, auc, roc_curve

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold

# For hyper-parameter-tuning
from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFE

# For pre-processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Importing ML models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')

sns.set_style('whitegrid', {'axes.grid' :False})
pio.templates.default = 'plotly_white'

# DATA ANALYSIS and PRE-PROCESSING
def explore_data(dset):
    print('Number of INSTANCES and ATTRIBUTES:', dset.shape)
    print('\n')
    print('Dataset COLUMNS:', dset.columns)
    print('\n')
    print('DATA TYPES of each columns:', dset.info()) 

def checking_removing_duplicates(dset):
    count_dups = dset.duplicated().sum()
    print('Number of DUPLICATES:', count_dups)
    if count_dups >= 1:
        dset.drop_duplicates(inplace=True)
        print('DUPLICATE values removed!')
    else:
        print('No DUPLICATE values!')
    
def read_in_and_split_data(data, target):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    return X_train, X_test, y_train, y_test

def GetModel():
    Models = []
    Models.append(('LR'   , LogisticRegression()))
    Models.append(('LDA'  , LinearDiscriminantAnalysis()))
    Models.append(('KNN'  , KNeighborsClassifier()))
    Models.append(('CART' , DecisionTreeClassifier()))
    Models.append(('NB'   , GaussianNB()))
    Models.append(('SVM'  , SVC(probability=True)))

    return Models

def ensemblemodels():
    ensembles = []
    ensembles.append(('AB'   , AdaBoostClassifier()))
    ensembles.append(('GBM'  , GradientBoostingClassifier()))
    ensembles.append(('RF'   , RandomForestClassifier()))
    ensembles.append(( 'Bagging' , BaggingClassifier()))
    ensembles.append(('ET', ExtraTreesClassifier()))
    
    return ensembles

def NormalizedModel(nameOfScaler):
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler == 'minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'normalizer':
        scaler = Normalizer()
    elif nameOfScaler == 'binarizer':
        scaler = Binarizer()

    pipelines = []
    pipelines.append((nameOfScaler+'LR', Pipeline([('Scaler', scaler), ('LR', LogisticRegression())])))
    pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    pipelines.append((nameOfScaler+'KNN' , Pipeline([('Scaler', scaler),('KNN' , KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', Pipeline([('Scaler', scaler),('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    pipelines.append((nameOfScaler+'SVM' , Pipeline([('Scaler', scaler),('SVM' , SVC())])))
    pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])))
    pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])))
    pipelines.append((nameOfScaler+'RF'  , Pipeline([('Scaler', scaler),('RF'  , RandomForestClassifier())])))
    pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])))

    return pipelines

def fit_model(X_train, y_train, Models):
    num_folds = 10
    scoring = 'accuracy'

    results = []
    names = []

    for name, model in Models:
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        
        results.append(cv_results)
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())

    return names, results

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))

def classification_metrics(model, conf_matrix):
    print(f"Training Accuracy Score: {model.score(X_train, y_train) * 100:.1f}%")
    print(f"Validation Accuracy Score: {model.score(X_test, y_test) * 100:.1f}%")
    
    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(pd.DataFrame(conf_matrix), annot = True, cmap = 'YlGnBu',fmt = 'g')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    plt.title('Confusion Matrix', fontsize=20, y=1.1)
    plt.ylabel('Actual label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.show()
    
    print(classification_report(y_test, y_pred1))

# Dataset
dset = pd.read_csv('Crop_recommendation.csv')
numerical_cols = dset.select_dtypes(include=[np.number])

# Outlier(s) removal
q1 = numerical_cols.quantile(0.25)
q3 = numerical_cols.quantile(0.75)
inter_q_range = q3 - q1
outlier_mask = ~((numerical_cols < (q1 - 1.5 * inter_q_range)) | (numerical_cols > (q3 + 1.5 * inter_q_range)))

outlier_data = dset[outlier_mask.all(axis=1)]

target = 'label'
X_train, X_test, y_train, y_test = read_in_and_split_data(dset, target)

# Training/Fitting the models

pipeline1 = make_pipeline(StandardScaler(), GaussianNB())
model1 = pipeline1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
matrix1 = confusion_matrix(y_test, y_pred1)
classification_metrics(pipeline1, matrix1)

pipeline2 = make_pipeline(StandardScaler(), SVC())
model2 = pipeline2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
matrix2 = confusion_matrix(y_test, y_pred2)
classification_metrics(pipeline2, matrix2)

pipeline3 = make_pipeline(StandardScaler(), DecisionTreeClassifier())
model3 = pipeline3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
matrix3 = confusion_matrix(y_test, y_pred3)
classification_metrics(pipeline3, matrix3)

pipeline4 = make_pipeline(StandardScaler(), LogisticRegression())
model4 = pipeline4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
matrix4 = confusion_matrix(y_test, y_pred4)
classification_metrics(pipeline4, matrix4)

pipeline5 = make_pipeline(StandardScaler(), KNeighborsClassifier())
model5 = pipeline5.fit(X_train, y_train)
y_pred5 = model5.predict(X_test)
matrix5 = confusion_matrix(y_test, y_pred5)
classification_metrics(pipeline5, matrix5)

pipeline6 = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
model6 = pipeline6.fit(X_train, y_train)
y_pred6 = model6.predict(X_test)
matrix6 = confusion_matrix(y_test, y_pred6)
classification_metrics(pipeline6, matrix6)

Models = GetModel()
print(fit_model(X_train, y_train, Models))

save_model(model1, 'model.pkl') # 1st Best Model
save_model(model3, 'model_best_2nd.pkl') # 2nd Best Model
save_model(model2, 'model_best_3rd.pkl') # 3rd Best Model