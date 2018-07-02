import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. Read data from file
wine = pd.read_csv('../data/wine_data.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])

#print(wine.head())

# 2. Set up data and label
X = wine.drop('Cultivator', axis=1)
y = wine['Cultivator']


# 3. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=123,
                                                    stratify=y)

# 4. Data preprocessing
scaler = StandardScaler()
# fit only to training data
scaler.fit(X_train)
# apply the transformations to data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 5. Training model
mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, y_train)

# 6. Predictions and Evaluation
predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))

