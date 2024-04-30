import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

#Creating file for selected features
cols = ['id', 'journal', 'key', 'number', 'volume', 'year']
data = pd.read_csv('selected_features.csv', sep = ";", error_bad_lines = False)
data.to_csv("selected_features.csv")

