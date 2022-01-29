import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Loading .csv file
df = pd.read_csv("iris.csv")

print(df.head())

# Select independent and dependent variables
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
Y = df["Class"]

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# Feature scaling 
sc = StandardScaler()
X_train =  sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file 
pickle.dump(classifier, open("model.pkl", "wb"))