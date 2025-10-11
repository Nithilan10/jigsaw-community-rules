import pandas as pd
from zipfile import ZipFile

zip_file_path = '../data/jigsaw-agile-community-rules.zip'
file_to_read = 'train.csv'

with ZipFile(zip_file_path, 'r') as z:
    with z.open(file_to_read) as f:
        df = pd.read_csv(f)

print(" Data loaded successfully!\n")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head)
