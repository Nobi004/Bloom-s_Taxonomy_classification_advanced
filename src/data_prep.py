import pandas as pd   
from io import StringIO 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os 

df = pd.read_csv("blooms.csv")

print(df.head())

df['text'] = df['text'].str.strip().str.lower().str.replace('"','')

label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

train, test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
val, test = train_test_split(test, test_size=0.5, stratify=test['label'], random_state=42)

os.makedirs('artifacts/data', exist_ok=True)
train.to_csv('artifacts/data/train.csv', index=False)
val.to_csv('artifacts/data/val.csv', index=False)
test.to_csv('artifacts/data/test.csv', index=False)

print("Data prepared and saved to artifacts/data/")