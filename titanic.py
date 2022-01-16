import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

import warnings
warnings.filterwarnings('ignore')

from google_drive_downloader import GoogleDriveDownloader
GoogleDriveDownloader.download_file_from_google_drive(file_id="*myfileid*",dest_path="./train.csv", unzip=False)
GoogleDriveDownloader.download_file_from_google_drive(file_id="*myfileid*",dest_path="./test.csv", unzip=False)
GoogleDriveDownloader.download_file_from_google_drive(file_id="*myfileid*",dest_path="./gender_submission.csv", unzip=False)

df = pd.read_csv('train.csv', sep=',')
df_test = pd.read_csv('test.csv', sep=',')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

data = pd.concat([train, test], sort=False)
train['train_or_test'] = 'train'
test['train_or_test'] = 'test'
data = pd.concat(
    [
        train,
        test
    ],
    sort=False
).reset_index(drop=True)

# Fare補完
fare = pd.concat([train['Fare'], test['Fare']])
# FAREは平均値でなく中央値で補完
data['Fare'].fillna(fare.mode()[0], inplace=True)
data.isnull().sum() 

#FAMILYを作る場合
# Familysize
data['FamilySize'] = data['Parch'] + data['SibSp'] + 1 #ALLデータ
# FamilySizeを離散化
data['FamilySize_bin'] = 'big'
data.loc[data['FamilySize']==1,'FamilySize_bin'] = 'alone'
data.loc[(data['FamilySize']>=2) & (data['FamilySize']<=4),'FamilySize_bin'] = 'small'
data.loc[(data['FamilySize']>=5) & (data['FamilySize']<=7),'FamilySize_bin'] = 'mediam'

#消さない場合
#Ticket
#data.loc[:, 'TicketFreq'] = data.groupby(['Ticket'])['PassengerId'].transform('count')

#Name
# テストデータの敬称(honorific)を抽出
data['honorific'] = data['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
# 敬称(honorific)の加工
data['honorific'].replace(['Col','Dr', 'Rev'], 'Rare',inplace=True) #少数派の敬称を統合
data['honorific'].replace('Mlle', 'Miss',inplace=True) #Missに統合
data['honorific'].replace('Ms', 'Miss',inplace=True) #Missに統合

#FareのEDA
f,ax=plt.subplots(1,2,figsize=(18,8), facecolor='gray')
data['Fare'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Fare')
ax[0].set_ylabel('')
sns.countplot('Fare',data=df,ax=ax[1])
ax[1].set_title('Fare')
plt.show()

#外れ値に対してロバストなモデルを構築したいのでビニング
data.loc[:, 'Fare_bin'] = pd.qcut(data.Fare, 14)

#Carbin
data['Cabin_ini'] = data['Cabin'].map(lambda x:str(x)[0])
data['Cabin_ini'].replace(['G','T'], 'Rare',inplace=True) #少数派のCabin_iniを統合
data.Embarked.fillna(df.Embarked.mode()[0], inplace=True)

#Pcalssの整数型を文字列にする
data.Pclass = data.Pclass.astype('str')

# カテゴリ特徴量についてlabel encoding
from sklearn import preprocessing
le_target_col = ['Sex', 'Fare_bin']
le = preprocessing.LabelEncoder()
for col in le_target_col:
    data.loc[:, col] = le.fit_transform(data[col])
    
#カテゴリカル変数をOne-Hot Encoding
cat_col = ['Embarked','FamilySize_bin', 'Pclass','Cabin_ini', 'honorific', 'Fare_bin']
data=pd.get_dummies(data, drop_first=True, columns=cat_col)

#相関性がありそうなのとかを捨てる
delete_columns = ['Name', 'PassengerId', 'Ticket', 'Age', 'Fare','Parch','SibSp','FamilySize','Cabin']
data.drop(delete_columns, axis=1, inplace=True)

from sklearn.model_selection import train_test_split

train = data.query('train_or_test == "train"')
test = data.query('train_or_test == "test"')

train.drop('train_or_test', axis=1, inplace=True)
test.drop('train_or_test', axis=1, inplace=True)


y_train = train['Perished']
x_train = train.drop('Perished', axis=1)
x_test = test.drop('Perished', axis=1)

# trainデータを分割
X_train, X_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.3, random_state=0)
y_train.isnull().sum()    


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rfc.score(X_train, y_train)}')
print(f'accuracy of test set: {rfc.score(X_test, y_test)}')

xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, y_train)
print('='*20)
print('XGBClassifier')
print(f'accuracy of train set: {xgb.score(X_train, y_train)}')
print(f'accuracy of train set: {xgb.score(X_test, y_test)}')

lgb = LGBMClassifier(random_state=0)
lgb.fit(X_train, y_train)
print('='*20)
print('LGBMClassifier')
print(f'accuracy of train set: {lgb.score(X_train, y_train)}')
print(f'accuracy of train set: {lgb.score(X_test, y_test)}')

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
print('='*20)
print('LogisticRegression')
print(f'accuracy of train set: {lr.score(X_train, y_train)}')
print(f'accuracy of train set: {lr.score(X_test, y_test)}')

svc = SVC(random_state=0)
svc.fit(X_train, y_train)
print('='*20)
print('SVC')
print(f'accuracy of train set: {svc.score(X_train, y_train)}')
print(f'accuracy of train set: {svc.score(X_test, y_test)}')


train = data.query('train_or_test == "train"')
test = data.query('train_or_test == "test"')

train.drop('train_or_test', axis=1, inplace=True)
test.drop('train_or_test', axis=1, inplace=True)


y_train = train['Perished']
x_train = train.drop('Perished', axis=1)
x_test = test.drop('Perished', axis=1)

!pip install optuna
from sklearn.model_selection import StratifiedKFold, cross_validate
import optuna

# CV分割数
cv = 5

# 上の学習の精度を見比べる感じこの前処理だと、ランダムフォレストが一番聞きそうなのでこれをチューニング
def objective(trial):
    
    param_grid_rfc = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        'min_samples_split': trial.suggest_int("min_samples_split", 7, 15),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_features': trial.suggest_int("max_features", 3, 10),
        "random_state": 0
    }

    model = RandomForestClassifier(**param_grid_rfc)
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=x_train, y=y_train, cv=kf)
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
rfc_best_param = study.best_params

best_clf = RandomForestClassifier(**rfc_best_param)
best_clf.fit(x_train, y_train)
clf_pred = best_clf.predict(x_test)

submission = pd.read_csv('gender_submission.csv')
clf_pred[:10]

submission['Perished'] = list(map(int, clf_pred))
submission.to_csv('submission.csv',index=False)
from google.colab import files
files.download('submission.csv')

