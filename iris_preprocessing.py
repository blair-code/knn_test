import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('./iris.data', header=None)
le = preprocessing.LabelEncoder()
le.fit(df.ix[:,4])
df.ix[:,4] = le.transform(df.ix[:,4])
df.to_csv('./iris.data.transf', header=False, index=False)