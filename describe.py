import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv')
args = parser.parse_args()

file = args.csv

data = pd.read_csv(file)

print(data.describe())
