import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('hepatitis.csv')

df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

class_le = LabelEncoder()
# M -> 1 and B -> 0

df.rename(columns={'class': 'classification'}, inplace=True)
df['sex'] = class_le.fit_transform(df.sex.values)
df['steroid'] = class_le.fit_transform(df.steroid.values)
df['fatigue'] = class_le.fit_transform(df.fatigue.values)
df['malaise'] = class_le.fit_transform(df.malaise.values)
df['anorexia'] = class_le.fit_transform(df.anorexia.values)
df['liver_big'] = class_le.fit_transform(df.liver_big.values)
df['liver_firm'] = class_le.fit_transform(df.liver_firm.values)
df['spleen_palpable'] = class_le.fit_transform(df.spleen_palpable.values)
df['spiders'] = class_le.fit_transform(df.spiders.values)
df['ascites'] = class_le.fit_transform(df.ascites.values)
df['varices'] = class_le.fit_transform(df.varices.values)
df['antivirals'] = class_le.fit_transform(df.antivirals.values)
df['classification'] = class_le.fit_transform(df.classification.values)
df['histology'] = class_le.fit_transform(df.histology.values)

train, test = train_test_split(df, test_size=0.2)

train_feature_names = train.columns[:-1]
train_feat = train[train_feature_names]
train_tar = train.classification

test_feature_names = test.columns[:-1]
test_feat = test[test_feature_names]
test_tar = test.classification

rf = RandomForestClassifier()

n_estimators = [100, 300]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf)

gridF = GridSearchCV(rf, hyperF, cv=3, verbose=1,
                     n_jobs=-1)

bestValues = gridF.fit(train_feat, train_tar)

print(bestValues)
