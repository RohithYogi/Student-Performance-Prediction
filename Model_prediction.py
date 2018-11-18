#!/usr/bin/env python
# coding: utf-8

# # Student Performance Analysis Model

# # Attributes
# 
# 1 school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# 
# 2 sex - student's sex (binary: 'F' - female or 'M' - male)
# 
# 3 age - student's age (numeric: from 15 to 22)
# 
# 4 address - student's home address type (binary: 'U' - urban or 'R' - rural)
# 
# 5 famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# 
# 6 Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# 
# 7 Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
# 
# 8 Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)
# 
# 9 Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# 10 Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# 
# 11 reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# 12 guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# 
# 13 traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 
# 14 studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 
# 15 failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 
# 16 schoolsup - extra educational support (binary: yes or no)
# 
# 17 famsup - family educational support (binary: yes or no)
# 
# 18 paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 
# 19 activities - extra-curricular activities (binary: yes or no)
# 
# 20 nursery - attended nursery school (binary: yes or no)
# 
# 21 higher - wants to take higher education (binary: yes or no)
# 
# 22 internet - Internet access at home (binary: yes or no)
# 
# 23 romantic - with a romantic relationship (binary: yes or no)
# 
# 24 famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 
# 25 freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 
# 26 goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# 27 Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# 28 Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# 29 health - current health status (numeric: from 1 - very bad to 5 - very good)
# 
# 30 absences - number of school absences (numeric: from 0 to 93)
# 
# 
# # Grades
# 
# 31 G1 - first period grade (numeric: from 0 to 20)
# 
# 31 G2 - second period grade (numeric: from 0 to 20)
# 
# 32 G3 - final grade (numeric: from 0 to 20, output target)
# 

# In[291]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import f1_score, recall_score, classification_report
from sklearn.metrics import fbeta_score
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from itertools import cycle
import pickle


# In[292]:


train1 = pd.read_csv('input/features.csv')
train1.head()


# # Correlation Plot

# In[293]:


def correlation(df):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(20, 15))
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.savefig('Correlation.png', bbox_inches='tight')
    plt.show()


# In[294]:


correlation(train1)


# In[295]:


from pandas.plotting import scatter_matrix
grades = train1[['G1','G2','G3']]
scatter_matrix(grades)
plt.savefig('grades.png', bbox_inches='tight')
plt.show()


# # One Hot Encoding on Final Grade

# In[296]:


le=preprocessing.LabelEncoder()


# In[297]:


le.fit(train1['FinalGrade'])
train1['FinalGrade']=le.transform(train1['FinalGrade'])
y=train1['FinalGrade']
# train1 = train1.drop(labels=['Regularity','Grade1','Grade2'],axis=1)


# In[298]:


train1 = pd.get_dummies(train1)


# In[299]:


train1.head(10)


# # Feature Drop

# In[300]:


# y=train1.FinalGrade
train1 = train1.drop(labels=['G3','FinalGrade','Fjob_at_home','Fjob_teacher','Pstatus_A','Pstatus_T'],axis=1)
train1.head()


# # SPLIT DATA 

# In[301]:


x_train,x_val,y_train,y_val = train_test_split(train1,y,random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


# # Confusion Matrix

# In[302]:


def confusionmatrix(y_val,y_pred):
    labels = list(range(0,5))
    cm=confusion_matrix(y_val,y_pred)
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); 
    ax.yaxis.set_ticklabels(labels);
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    total = lambda x : x.sum()/5
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('percentage of sensitivity = '+str(total(TPR)*100))

    # Specificity or true negative rate
    TNR = TN/(TN+FP) 

    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('percentage of precision = '+str(total(PPV)*100))
    # Negative predictive value
    NPV = TN/(TN+FN)

    # Fall out or false positive rate
    FPR = FP/(FP+TN)

    # False negative rate
    FNR = FN/(TP+FN)

    # False discovery rate
    FDR = FP/(TP+FP)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Accuracy percentage = '+str(total(ACC)*100))


# # ROC plot

# In[303]:


def ROC_plot(x_train,x_val,model):
    train = pd.read_csv('features.csv')
    train.head()
    y=train[['FinalGrade']]
    train = train.drop(['G3'],axis=1);
    train = train.drop(['FinalGrade'],axis=1);
    train = train.drop(['G2'],axis=1);
    train = train.drop(['G1'],axis=1);
    y = label_binarize(y, classes=['Failure','Poor','Satisfactory','Good','Excellent'])
    n_classes = y.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(train,y,random_state=0)
    
    classifier = OneVsRestClassifier(model)
    y_score = classifier.fit(x_train, y_train).decision_function(x_val)
    y_score.shape
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    # Individual ROC
    plt.figure()
    lw = 2
    for i in (0,1):
        plt.subplot(1,2,i+1)
        plt.plot(fpr[i], tpr[i], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic label'+str(i))
        plt.legend(loc="lower right")
    plt.savefig('ROC1.png', bbox_inches='tight')
    plt.plot()

    plt.figure()
    lw = 2
    for i in (2,3):
        plt.subplot(1,2,i-1)
        plt.plot(fpr[i], tpr[i], color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic label'+str(i))
        plt.legend(loc="lower right")
    plt.savefig('ROC2.png', bbox_inches='tight')
    plt.plot()

    plt.figure()
    lw = 2
    plt.subplot(1,2,1)
    plt.plot(fpr[4], tpr[4], color='grey',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[4])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic label'+str(4))
    plt.legend(loc="lower right")
    plt.savefig('ROC3',box_inches='tight')
    plt.plot()
    
    
    # Combined ROC
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
        print('Area Under the Curve with label '+str(i)+' is '+str(roc_auc[i]))
    plt.savefig('ROC4', bbox_inches='tight')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.rcParams["figure.figsize"] = (10,6)


# # Fscore

# In[304]:


def Fscore(y_val,y_pred):
    print('f score = ' + str(f1_score(y_val, y_pred, average="macro")))


# # Recall

# In[305]:


def recall(y_val,y_pred):
    print('percentage of recall score = '+str(recall_score(y_val, y_pred, average="macro"))) 


# # Classification Report

# In[306]:


def report(y_val,y_pred):
    target_names = ['Failure','Poor','Satisfactory','Good','Excellent']
    print('Classification Report')
    print(classification_report(y_val, y_pred, target_names=target_names))


# # F Beta score

# In[307]:


def fbeta(y_val,y_pred):
    print('Fbeta score = ' + str(fbeta_score(y_val,y_pred,average='macro', beta=0.5)))


# # LOGISTIC REGRESSION 

# In[308]:


def logistic_regression_model(x_train,y_train,x_val,y_val):
    lr =  LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_val)
    y_predict = lr.predict_proba(x_val)
    print("Log_Loss: ",log_loss(y_val,y_predict))
    print("Accuracy_Score: ",accuracy_score(y_val,y_pred))
    confusionmatrix(y_val,y_pred)
    Fscore(y_val,y_pred)
    recall(y_val,y_pred)
    report(y_val,y_pred)
    fbeta(y_val,y_pred)
    return lr


# In[309]:


model =logistic_regression_model(x_train,y_train,x_val,y_val)
ROC_plot(x_train,x_val,model)


# In[310]:


filename = 'pickle/model_lr.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# # RANDOM FOREST

# In[311]:


def random_forest_model(x_train,y_train,x_val,y_val):
    random_forest = RandomForestClassifier(n_estimators=28,max_depth=5,random_state=0)

    forest = random_forest.fit(x_train, y_train)
    print("Random Forest Train data Score" , ":" , forest.score(x_train, y_train) 
          , "," ,"Validation data Score" ,":" , forest.score(x_val, y_val))
    Y_pred = random_forest.predict_proba(x_val)
    Y_pred1 = random_forest.predict(x_val)
    print("Log_Loss: ",log_loss(y_val,Y_pred))
    confusionmatrix(y_val,Y_pred1)
    Fscore(y_val,Y_pred1)
    recall(y_val,Y_pred1)
    report(y_val,Y_pred1)
    fbeta(y_val,Y_pred1)
    return forest


# In[312]:


model = random_forest_model(x_train,y_train,x_val,y_val)


# In[313]:


filename = 'pickle/model_rf.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# # SVM

# In[314]:


def SVM_Model(X_train,Y_train,X_test,y_val):
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    print("SVM Train data Score" , ":" , svc.score(X_train, y_train)
          , "," ,"Validation data Score" ,":" , svc.score(X_test, y_val))
    confusionmatrix(y_val,Y_pred)
    Fscore(y_val,Y_pred)
    recall(y_val,Y_pred)
    report(y_val,Y_pred)
    fbeta(y_val,Y_pred)
    return svc


# In[315]:


model = SVM_Model(x_train,y_train,x_val,y_val)
ROC_plot(x_train,x_val,model)


# In[316]:


filename = 'pickle/model_svm.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# # DECISION TREE

# In[317]:


def Decison_tree_Model(x_train,y_train,x_val,y_val):
    tree = DecisionTreeClassifier(min_samples_leaf=9,random_state=0)
    tf= tree.fit(x_train, y_train)
    y_pred = tf.predict(x_val)
    y_predict = tf.predict_proba(x_val)
    print("Decisioin Tree Train data Score" , ":" , tf.score(x_train, y_train) 
          , "," , "Validation data Score" ,":" , tf.score(x_val, y_val))
    confusionmatrix(y_val,y_pred)
    print("Log_Loss: ",log_loss(y_val,y_predict))
    Fscore(y_val,y_pred)
    recall(y_val,y_pred)
    report(y_val,y_pred)
    fbeta(y_val,y_pred)
    return tree


# In[318]:


model = Decison_tree_Model(x_train,y_train,x_val,y_val)


# In[319]:


filename = 'pickle/model_dt.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# # ADA BOOST

# In[320]:


def ada_boost_model(x_train,y_train,x_val,y_val):
    ada = AdaBoostClassifier(n_estimators=2)
    af = ada.fit(x_train, y_train)
    y_pred = af.predict(x_val)
    y_predict = af.predict_proba(x_val)
    print("Ada Boost Train data Score" , ":" , af.score(x_train, y_train) 
          , "," ,"Validation data Score" ,":" , af.score(x_val, y_val))
    print("Log_Loss: ",log_loss(y_val,y_predict))
    confusionmatrix(y_val,y_pred)
    Fscore(y_val,y_pred)
    recall(y_val,y_pred)
    report(y_val,y_pred)
    fbeta(y_val,y_pred)
    return ada


# In[321]:


model = ada_boost_model(x_train,y_train,x_val,y_val)
ROC_plot(x_train,x_val,model)


# In[322]:


filename = 'pickle/model_ada.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# # XGBOOST

# In[323]:


def XGBoost(x_train,y_train,x_val,y_val):
    model = XGBClassifier()
    model = XGBClassifier(learning_rate=0.1,n_estimators=80)
    mf = model.fit(x_train,y_train)
    y_pred=model.predict(x_val)
    y_predict = mf.predict_proba(x_val)
    print("XGBoost Train data Score" , ":" , mf.score(x_train, y_train) 
          , "," ,"Validation data Score" ,":" , mf.score(x_val, y_val))
    print("Log_Loss: ",log_loss(y_val,y_predict))
    confusionmatrix(y_val,y_pred)
    Fscore(y_val,y_pred)
    recall(y_val,y_pred)
    report(y_val,y_pred)
    fbeta(y_val,y_pred)
    
    
    # plot feature importance
    fig, ax = plt.subplots(figsize=(10, 20))
    plot_importance(model, ax=ax)
    plt.savefig('Feature_Engineering.png', bbox_inches='tight')
    plt.show()
    return model
    
    


# In[324]:


model = XGBoost(x_train,y_train,x_val,y_val)


# # K Cross Validations

# In[325]:


def k_cross_validations(x_train,y_train,):
    X = x_train
    y = y_train
    kf = KFold(n_splits=10) # Define the split - into 2 folds 
    kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
    print(kf) 
    KFold(n_splits=10, random_state=None, shuffle=False)
    return kf


# In[326]:


kf = k_cross_validations(x_train,y_train)
classifier = model
cross_val_score(classifier,x_train, y_train, cv=kf, n_jobs=1)


# In[327]:


def FeatureImportance():
    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000,
                               n_features=10,
                               n_informative=3,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=0,
                               shuffle=False)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


# In[328]:


FeatureImportance()


# # Finally we choose XGBoost Model
# 
# # Train data Score : 0.9386973180076629 
# 
# # Validation data Score : 0.8850574712643678

# In[329]:


filename = 'pickle/model_xgb.pkl'
outfile = open(filename,'wb')
pickle.dump(model,outfile)
outfile.close()


# In[ ]:




