import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from pandas import DataFrame

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    print(metric.__name__, "WITHOUT preparation:", metric(y_pred, y, average='weighted'))
    print(metric.__name__, "WITH preparation:", metric(y_prep_pred, y_prep, average='weighted'))

df = pd.read_csv('datasets/TotalFeatures-ISCXFlowMeter.csv')

df.head(10)

df.info()

df['calss'].value_counts()

X = df.copy()
X['calss'] = X['calss'].factorize()[0]

corr_matrix = X.corr()
corr_matrix["calss"].sort_values(ascending=False)

X.corr()

corr_matrix[corr_matrix["calss"] > 0.05]

train_set, val_set, test_set = train_val_test_split(X)

X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)

scaler = RobustScaler()
X_test_scaled = scaler.fit_transform(X_test)

scaler = RobustScaler()
X_val_scaled = scaler.fit_transform(X_val)

X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_train_scaled.head(10)

X_train_scaled.describe()

from sklearn.tree import DecisionTreeClassifier

MAX_DEPTH = 20


clf_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
clf_tree.fit(X_train, y_train)

clf_tree_scaled = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
clf_tree_scaled.fit(X_train_scaled, y_train)

y_train_pred = clf_tree.predict(X_train)
y_train_prep_pred = clf_tree_scaled.predict(X_train_scaled)

evaluate_result(y_train_pred, y_train, y_train_prep_pred, y_train, f1_score)

y_pred = clf_tree.predict(X_val)
y_prep_pred = clf_tree_scaled.predict(X_val_scaled)

evaluate_result(y_pred, y_val, y_prep_pred, y_val, f1_score)

clf_tree_reduced = DecisionTreeClassifier(max_depth=2, random_state=42)
clf_tree_reduced.fit(X_train_reduced, y_train)

from graphviz import Source
from sklearn.tree import export_graphviz
import os

export_graphviz(
        clf_tree_reduced,
        out_file="android_malware.dot",
        feature_names=X_train_reduced.columns,
        class_names=["benign", "adware", "malware"],
        rounded=True,
        filled=True
    )

Source.from_file("android_malware.dot")