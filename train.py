import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set random seed
seed = 22092020
results_dir = Path('results/metrics.txt')
results_dir.parent.mkdir(exist_ok=True, parents=True)

################################
########## DATA PREP ###########
################################

# Load in the data
# df = pd.read_csv("wine_quality.csv")
df = pd.read_csv("data/winequalityN.csv")
df = df.dropna()

le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
mapping = dict(zip(le.classes_, range(len(le.classes_))))


# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=seed)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(n_estimators=80, max_depth=15, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
train_score = regr.score(X_train, y_train) * 100
# Report test set score
test_score = regr.score(X_test, y_test) * 100

metrics_scores = {
    'train': round(train_score, 3),
    'test': round(test_score, 3)
}

# Write scores to a file
with open("results/metrics.txt", 'w') as outfile:
    outfile.write(json.dumps(metrics_scores))


##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

sns.barplot(x="importance", y="feature", data=feature_df, ax=ax[0])
ax[0].set_xlabel('Importance',fontsize = axis_fs) 
ax[0].set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax[0].set_title('RF - Feature importance', fontsize = title_fs)

# plt.tight_layout()
# plt.savefig("results/feature_importance.png",dpi=120) 
# plt.close()


##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = regr.predict(X_test) + np.random.normal(0,0.05,len(y_test))
y_jitter = y_test + np.random.normal(0,0.05,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

sns.scatterplot(x="true", y="pred",data=res_df, ax=ax[1])
ax[1].set_aspect('equal')
ax[1].set_xlabel('True wine quality',fontsize = axis_fs) 
ax[1].set_ylabel('Predicted wine quality', fontsize = axis_fs)#ylabel
ax[1].set_title('RF - Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax[1].plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("results/fi_and_residuals.png",dpi=120) 

