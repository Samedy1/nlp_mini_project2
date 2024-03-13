from preprocess import *
import os
from sklearn.metrics import accuracy_score
import pickle

max_accuracy = 0
X_test = test_df[input_features]
y_test = test_df['sentiment']

# import model
for file in os.listdir('exported_models'):
    with open(f'exported_models/{file}', 'rb') as f:
        loaded_model = pickle.load(f)
    y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if (max_accuracy < accuracy):
        max_accuracy = accuracy
        best_model = loaded_model

model = best_model

print(model)

with open('challenge/challenge_data.txt') as infile:
    challenge_data = infile.read()

lst = challenge_data.split('\n')

y_pred = model.predict(preprocess(lst)[input_features])

result = ''

for r in y_pred:
    result = f'{result}{r}'

with open('challenge/group10_mini_project_2_challenge.txt', 'w') as outfile:
    outfile.write(result)