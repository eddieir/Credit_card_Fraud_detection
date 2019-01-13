import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc
from sklearn.model_selection import train_test_split

data =pd.read_csv('creditcard.csv')
print(data.head())

data[data['Class']==1].describe()    # Describing the data

# Scaling the Time and Amount features
data['Scaled_Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data['Scaled_Time'] = StandardScaler().fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Time', 'Amount'], axis=1, inplace=True)
data.head()

# Splitting the data into input features (X), and output target (Y)
X = data.iloc[:, data.columns != "Class"]
Y = data.iloc[:, data.columns == "Class"]

# Splitting the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=50)


# Decision Tree

clf= DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(classification_report(y_test, pred))

matrix = confusion_matrix(y_test, pred)
print(matrix)



sns.heatmap(matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()


# Calculating the Area Under the Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)



# Plotting the ROC Curve
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Finding the important features from the data
clf = RandomForestClassifier()
clf.fit(X, Y)
important_feat = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(important_feat)

# Plotting the important features with respect to their importance
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(data.columns, clf.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)
plt.show()

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X[['V4','V9','V10','V11','V12','V14','V16','V17','V18']], Y,
                                                    test_size=0.30, random_state=50)

X_train_new.head()

y_train_new.head()

clf_new= DecisionTreeClassifier()
clf_new.fit(X_train_new, y_train_new)
pred_new = clf_new.predict(X_test_new)

print(classification_report(y_test_new, pred_new))

matrix_new = confusion_matrix(y_test_new, pred_new)
print(matrix_new)

sns.heatmap(matrix_new, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_new, pred_new)
roc_auc_lr_new = auc(false_positive_rate, true_positive_rate)
print (roc_auc_lr_new)

plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc_lr_new)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

clf_new= RandomForestClassifier()
clf_new.fit(X_train_new, y_train_new)
pred_new = clf_new.predict(X_test_new)

print(classification_report(y_test_new, pred_new))

matrix_new = confusion_matrix(y_test_new, pred_new)
print(matrix_new)

sns.heatmap(matrix_new, cmap="coolwarm_r", annot=True, linewidths=0.5)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Real Class")
plt.show()

alse_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_new, pred_new)
roc_auc_lr_new = auc(false_positive_rate, true_positive_rate)
print (roc_auc_lr_new)

plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc_lr_new)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






