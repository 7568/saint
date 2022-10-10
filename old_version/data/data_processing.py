import os

import numpy as np
import pandas as pd

df = pd.read_csv(os.getcwd()+'/forest/covtype.data',header=None)
print(df.shape)
df = df[df[54]<=2]
df[54] = df[54] - 1
print(df.shape)
df = df.loc[:, df.std() > 0]
print(df.shape)
df.head()

df.to_csv(os.getcwd()+'/forest.csv',sep=',',index = False)

df = pd.read_csv(os.getcwd()+'/forest.csv',header=None,skiprows=1)
df.head()