import pandas as pd 
data = pd.read_csv("student_scores.csv")
print(data.head(10))

print('columns:', data.columns.tolist())
print('\nInfo:')
print(data.info())
print('\nDescribe:')
print(data.describe())
print(data.shape)
print(data.isnull().sum())
data["score"] = data["score"].fillna(data["score"].mean())
data = data.dropna()
print(data.isnull().sum())
print(data.head(10))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#data["gender"] = le.fit_transform(data["gender"])
data["gender"] = data["gender"].map({"M":1, "F":0})
print(data.head(20))


le = LabelEncoder()
data["class_encoded"] = le.fit_transform(data["grades"])

X = data[["hours","class_encoded"]]
Y = data["score"]
print(type(X))
print(type(X))