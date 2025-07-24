import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('/content/diabetes.csv')

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy on training data:', training_accuracy)

X_test_prediction = classifier.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data:', test_accuracy)

print("\nEnter the following values to check if the person is diabetic:")
features = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin',
            'BMI', 'Diabetes in Genetics', 'Age']
input_data = []

for feature in features:
    value = float(input(f"{feature}: "))
    input_data.append(value)

input_array = np.asarray(input_data).reshape(1, -1)

std_input = scaler.transform(input_array)

prediction = classifier.predict(std_input)

if prediction[0] == 0:
    print('\nResult: The person is NOT diabetic.')
else:
    print('\nResult: The person IS diabetic.')

