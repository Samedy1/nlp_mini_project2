from preprocess import *
from sklearn.ensemble import RandomForestClassifier

input_features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

X_train = training_df[input_features]
y_train = training_df['sentiment']

X_test = test_df[input_features]
y_test = test_df['sentiment']

# import model
model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

with open('challenge/challenge_data.txt') as infile:
    challenge_data = infile.read()

lst = challenge_data.split('\n')

y_pred = model.predict(preprocess(lst)[input_features])

result = ''

for r in y_pred:
    result = f'{result}{r}'

with open('challenge/group10_mini_project_2_challenge.txt', 'w') as outfile:
    outfile.write(result)