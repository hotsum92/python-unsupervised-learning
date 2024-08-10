import os
import pandas as pd

current_path = os.getcwd()
file = os.path.join(current_path, 'credit-card.csv')
data = pd.read_csv(file)

print(data.describe())
