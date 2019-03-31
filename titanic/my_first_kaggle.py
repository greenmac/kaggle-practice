import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
dataset = pd.read_csv(SCRIPT_PATH + "/train.csv")

#features = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
training_data = dataset[features]
training_label = dataset['Survived']

raining_data = training_data.fillna(0)
# print(raining_data)
#training_data['Name'] = LabelEncoder().fit_transform(training_data['Name']) # 不重要
training_data['Sex'] = LabelEncoder().fit_transform(training_data['Sex'])
#training_data['Ticket'] = LabelEncoder().fit_transform(training_data['Ticket']) # 不重要
# training_data['Cabin'] = LabelEncoder().fit_transform(training_data['Cabin'].astype(str))
# training_data['Embarked'] = LabelEncoder().fit_transform(training_data['Embarked'].astype(str))

# model = RandomForestClassifier()
# model.fit(training_data, training_label)

# print(training_data['Embarked'])

# y_pos = np.arange(len(features))
# plt.barh(y_pos, model.feature_importances_, align='center', alpha=0.4)
# plt.yticks(y_pos, features)
# plt.xlabel('features')
# plt.title('feature_importances')
# plt.show()