import os
import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing, svm, tree

filename = '190122_3_categories.xlsx'
current_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_path, filename)
print(file_path)

df = pd.read_excel(file_path, sheet_name=[0,1,2])  # 0->'sheet1', early stage; 1->suspect; 2->health

# early -> 1
# suspect -> -1
# health -> 0

early_list = df[0].values.tolist()
suspect_list = df[1].values.tolist()
health_list = df[2].values.tolist()

# early_list = [early[3:17] for early in early_list]  # 17
# suspect_list = [suspect[3:17] for suspect in suspect_list]  # 17
# health_list = [health[3:14] for health in health_list]  # 14

early_list = [early[3:] for early in early_list]  # 17
suspect_list = [suspect[3:] for suspect in suspect_list]  # 17
health_list = [health[3:] for health in health_list]  # 14


maxNCT = round(random.uniform(15, 21), 1)
max_min = round(random.uniform(0, 5), 1)
minNCT = maxNCT - max_min
for health in health_list:
    health.insert(11, maxNCT)
    health.insert(12, minNCT)
    health.insert(13, max_min)

assert len(early_list[0])==len(suspect_list[0])==len(health_list[0])

early_label = [1] * len(early_list)
suspect_label = [-1] * len(suspect_list)
health_label = [0] * len(health_list)

train_all = early_list + suspect_list + health_list
label_all = early_label + suspect_label + health_label

assert len(train_all) == len(label_all)
print(len(label_all))

# pre-processing
scaler = preprocessing.StandardScaler().fit(train_all)
train_all = scaler.transform(train_all)


# score = clf.score(train_all, label_all)
# print(score)

# scores = cross_val_score(clf, train_all, label_all, cv=5)
# print(scores)
# print(scores.mean())

def K_folder_LR():
    k_folder = 3
    skf = StratifiedKFold(n_splits=k_folder, shuffle=True) # improved from 0.8 to 0.9 after shuffle
    avg_score = 0
    for train_index, test_index in skf.split(train_all, label_all):
        print(test_index)
        X_train, X_test = np.array(train_all)[train_index], np.array(train_all)[test_index]
        y_train, y_test = np.array(label_all)[train_index], np.array(label_all)[test_index]
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        avg_score += score
    avg_score /= k_folder
    print(avg_score)


def K_folder_SVM():
    k_folder = 3
    skf = StratifiedKFold(n_splits=k_folder, shuffle=True)  # improved from 0.8 to 0.9 after shuffle
    avg_score = 0
    for train_index, test_index in skf.split(train_all, label_all):
        print(test_index)
        X_train, X_test = np.array(train_all)[train_index], np.array(train_all)[test_index]
        y_train, y_test = np.array(label_all)[train_index], np.array(label_all)[test_index]
        clf = svm.SVC(gamma='scale', decision_function_shape='ovo').fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        avg_score += score
    avg_score /= k_folder
    print(avg_score)

def K_folder_DT():
    k_folder = 3
    skf = StratifiedKFold(n_splits=k_folder, shuffle=True)  # improved from 0.8 to 0.9 after shuffle
    avg_score = 0
    for train_index, test_index in skf.split(train_all, label_all):
        print(test_index)
        X_train, X_test = np.array(train_all)[train_index], np.array(train_all)[test_index]
        y_train, y_test = np.array(label_all)[train_index], np.array(label_all)[test_index]
        clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        avg_score += score
    avg_score /= k_folder
    print(avg_score)


if __name__ == '__main__':
    # K_folder_LR()
    K_folder_DT()

