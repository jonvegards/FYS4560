#!/usr/bin/env python3

"""
Generate a tree-model by using GradientBoostingClassifier
from Scikit-learn.

- This script is handling files with header like:
    type, lepton.pt(), lepton.eta(), lepton.phi(), b_jet.pt(), b_jet.eta(), b_jet.phi(), etmiss.et(), etmiss.phi()
- Model can be saved and used later

@author Jon Vegard Sparre
May 2017
"""

import numpy as np
import matplotlib.pyplot as plt
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(**params)
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import ensemble
import textwrap, sys

print('Pandas version ' + pd.__version__)
print('Sklearn version ' + sk.__version__)

####################################################################
# Read in and treat data (mostly) using pandas (none were harmed!) #
####################################################################

# Read in data to train model
background = pd.read_csv('background_alljets_onebjet.txt', skipinitialspace=True)
signal     = pd.read_csv('signal_alljets_onebjet.txt', skipinitialspace=True)

# Create data frames in pandas
df_b = pd.DataFrame(background)
df_s = pd.DataFrame(signal)

# If you want to use a subset of the signal data
# df_s = df_s[:200000]

# Check properties of data
print("df_s:")
print("len(df_s) = {}".format(len(df_s)))
print("df_s.shape = {}".format(df_s.shape))
print("df_b:")
print("len(df_b) = {}".format(len(df_b)))
print("df_b.shape = {}".format(df_b.shape))

# Concatenate data frames
frames = [df_s, df_b]
df = pd.concat(frames, ignore_index=True)
print("Merged data frame")
print("len(data) = {}".format(len(df)))
print("data.shape = {}".format(df.shape))

# Split data for training using function from pandas
train = df.sample(frac=0.9, random_state=42, replace=False)
test  = df.drop(train.index)
print("Training and test data frames:")
print("len(data) = {} {}".format(len(train), len(test)))
print("data.shape = {} {}".format(train.shape, test.shape))

# Define features to train on and target numbers (used below)
# Convert to array for scikit using values
# ravel is used to avoid [[n]], i.e. n x 1 arrays
# Defining two feature_list, one with mass variables and one without
feature_list = ["lepton.pt()", "lepton.eta()", "lepton.phi()", "b_jet.pt()", "b_jet.eta()", "b_jet.phi()", "etmiss.et()", "etmiss.phi()", "TransMass", "InvMass"]
# feature_list = ["lepton.pt()", "lepton.eta()", "lepton.phi()", "b_jet.pt()", "b_jet.eta()", "b_jet.phi()", "etmiss.et()", "etmiss.phi()"]
print("Feature list: %s" %(feature_list))
n_features   = len(feature_list)
target_list  = ["type"]
features = train[feature_list].values
target   = train[target_list].values.ravel()
features_test = test[feature_list].values
target_test   = test[target_list].values.ravel()

###################################
# Random decision tree regression #
# with gradient boosting          #
###################################

"""
Note: depth of trees must be sufficient to recreate quantities of interest
given available features, e.g. invariant mass needs depth > 4
"""

# Define training parameters and call the classifier
params = {'n_estimators': 100, 'max_depth': 1, 'random_state': 42, 'learning_rate': 0.1, 'verbose' : 1,
'subsample' : .5}
# clf = ensemble.GradientBoostingClassifier(**params)
# clf.fit(features, target)

# Saving model
from sklearn.externals import joblib
# joblib.dump(clf, 'AllData_depth1_n_estimators100_TransInv.pkl')
# Load premade model with depth 1, 100 estimators and with transverse
# and invariant masses.
clf = joblib.load('AllData_depth1_n_estimators100_TransInv.pkl')

predicted = clf.predict(features_test)

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(target_test, predicted, target_names=["No top", "Top quark"])))

acc = clf.score(features_test, target_test)
print("Accuracy: {:.4f}".format(acc))

# Plot feature importance
feature_importance = clf.feature_importances_
# Scale importances relative to total
feature_importance = 100.0 * (feature_importance / feature_importance.sum())
sorted_importance = np.argsort(feature_importance)
feature_list = np.array(feature_list)
pos = np.arange(sorted_importance.shape[0]) + .5

plt.style.use('ggplot')

plt.figure(1)
plt.barh(pos, feature_importance[sorted_importance], align='center')
plt.yticks(pos, feature_list[sorted_importance])
plt.xlabel('Relative Importance')
plt.title("\n".join(textwrap.wrap('Variable importance, params = {}'.format(params), 80)),fontsize=11)
plt.tight_layout()
plt.savefig('var_imp_alljets_d1.pdf')


# Compute test and plot deviances
test_score = np.zeros( (params['n_estimators'],) )

# Use staged_decision_function for classification:
for i, y_pred in enumerate( clf.staged_decision_function(features_test) ):
    test_score[i] = clf.loss_(target_test, y_pred)

train_score = clf.train_score_

plt.figure(3)
plt.title("\n".join(textwrap.wrap('Deviance, params = {}'.format(params), 80)),fontsize=11)
plt.plot(np.arange(params['n_estimators']) + 1, test_score,
         label='Test Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, train_score,
         label='Training Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
plt.savefig('deviance_alljets_d1.pdf')
plt.show()