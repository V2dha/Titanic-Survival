import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
import scipy.optimize as opt
from sklearn import preprocessing
%matplotlib inline
import matplotlib.pyplot as plt

# %% [code]
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df1 = pd.read_csv("/kaggle/input/titanic/test.csv")
df1.head()

# %% [code]
train_x = np.asarray(df[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare']])
train_y = np.asarray(df[['Survived']])
train_x[0:5], train_y[0:5]
train_x.shape, train_y.shape

test_x = np.asarray(df1[['Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare']])
test_x[0:5]
test_x.shape


# %% [code]
from sklearn import preprocessing
Sex = preprocessing.LabelEncoder()
Sex.fit(['female','male'])
train_x[:,1] = Sex.transform(train_x[:,1])
test_x[:,1] = Sex.transform(test_x[:,1])
test_x[0:5], train_x[0:5]

# %% [code]
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
train_x = my_imputer.fit_transform(train_x)
test_x = my_imputer.fit_transform(test_x)

# %% [code]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR =  LogisticRegression(C=0.01, solver = 'sag').fit(train_x, train_y)
LR

# %% [code]
pred_y = LR.predict(test_x)
pred_y

# %% [code]
pred_yp = LR.predict_proba(test_x)
pred_yp


# %% [code]
pas = np.asarray(df1[['PassengerId']])
pas = np.concatenate(pas)
pas.shape, pred_y.shape

# %% [code]

dict = {'PassengerId': pas, 'Survived': pred_y}
sub = pd.DataFrame(dict)
sub.to_csv(r'Submission 3.csv', index=False) 