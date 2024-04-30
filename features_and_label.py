import pandas as pd
import numpy as np
import csv

#Loading data
data = pd.read_csv('selected_features.csv')
data.head()

#The classes we need to work on
from io import StringIO

#Creating Count and journal_class column
df = data.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('.'))
col = ['id', 'journal','year']
df = df[col]
df = df[pd.notnull(df['id'])]
df.columns = ['id', 'journal','year']
df['Count'] = df.groupby(['journal'])['id'].transform('count')
df['journal_class'] = df['journal'].factorize()[0]
Count_df = df[['id','journal', 'journal_class', 'Count', 'year']].drop_duplicates().sort_values('year')
df.head()
Count_df.to_csv("features_and_label.csv")

