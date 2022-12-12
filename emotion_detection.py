## Model Expermination for the Emotion Detectin with multiple methods ##

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import neattext.functions as nfx
from textblob import TextBlob
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
import nltk
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

#Method to create Tfidf Vectorizer
def initialize():
  #Dataset which is pre processed already with emotiondetectionpreprocess.py file
  df = pd.read_csv("balanced_filtered_tweet_emotions.csv")
  df.head()

  df.dropna(inplace=True)

  df.isna().sum().sum()
  #Shuffling the dataset before progressing
  cols_df = df.columns
  df = df.sample(frac=1).reset_index()
  df = df[cols_df]
  X = df["updated_content"]
  y = df["emotion"]
  #Using Tfidf Vectoriser 
  tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=2)
  train = tfidf.fit_transform(X)
  return train, y, tfidf

#Method which returns treu if the string contains digits
def containsNumber(value):
    for character in value:
        if character.isdigit():
            return True
    return False

#Removes numbered words from text
def remove_numbered_words(text):
  return ' '.join([token for token in text.split() if not containsNumber(token)])


#Method which preproesses the messages 
def preprocess(new_df):
  new_df["updated_message"] = new_df.message.apply(lambda x: x.lower())
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_punctuations)
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_userhandles)
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_stopwords)
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_urls)
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_numbers)
  new_df["updated_message"] = new_df.updated_message.apply(nfx.remove_accents)
  # new_df["updated_message"] = new_df.updated_message.apply(lambda x: str(TextBlob(x).correct()))
  # new_df["updated_message"] = new_df.updated_message.apply(remove_numbered_words)

  new_df["updated_message"] = new_df.updated_message.apply(stem_text)
  new_df["updated_message"] = new_df.updated_message.apply(lemmatize_text)
  return new_df

#Method which stems text
def stem_text(text):
  porter = PorterStemmer()
  return ' '.join([porter.stem(token) for token in text.split()]) 

#Method to lemmatize
def lemmatize_text(text):
  wordnet_lemmatizer = WordNetLemmatizer()
  return ' '.join([wordnet_lemmatizer.lemmatize(token) for token in text.split()])


#Method to experiment best number of features to extract from,
#using Select K Best method using chi2 for SVM Model
def experiment_SVM(val, train, y):
  #Feature Selection with Select K Best and Chi2 Test
  X_new = SelectKBest(chi2, k=val).fit_transform(train, y)
  X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 42, stratify=y)
  gs_svc = GridSearchCV(SVC(gamma='auto'), {
    'C': [1, 10, 20, 30, 40],
    'kernel': ['rbf','linear']
  }, cv=5, return_train_score=False)
  gs_svc.fit(X_train, y_train)
  gs_svc_df = pd.DataFrame(gs_svc.cv_results_)
  best_svc = gs_svc_df[gs_svc_df["mean_test_score"] == gs_svc_df["mean_test_score"].max()]
  svc_C_param = best_svc["param_C"]
  SVM_train_preds_fs = gs_svc.predict(X_train)
  accuracy_SVM_train_fs = accuracy_score(y_train, SVM_train_preds_fs)
  f1_svm_train_fs = f1_score(y_train, SVM_train_preds_fs, average='micro')
  SVM_test_preds_fs = gs_svc.predict(X_test)
  accuracy_SVM_test_fs = accuracy_score(y_test, SVM_test_preds_fs)
  f1_svm_test_fs = f1_score(y_train, SVM_train_preds_fs, average='micro')
  return accuracy_SVM_train_fs, accuracy_SVM_test_fs, f1_svm_train_fs, f1_svm_test_fs

#Method to experiment best number of features to extract from,
#using Select K Best method using chi2 for SVM Model
def experiment_NB(val, train, y):
  X_new = SelectKBest(chi2, k=val).fit_transform(train, y)
  X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 42, stratify=y)
  params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
          'fit_prior': [True, False],
          'binarize': [None, 0.0, 8.5, 10.0]
         }
  gs_nb = GridSearchCV(BernoulliNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
  gs_nb.fit(X_train,y_train)
  gs_nb_df = pd.DataFrame(gs_nb.cv_results_)
  best_nb = gs_nb_df[gs_nb_df["mean_test_score"] == gs_nb_df["mean_test_score"].max()]

  #Testing the NB Model with best parameters obtained
  NB_train_preds = gs_nb.predict(X_train)

  # print(confusion_matrix(train_target, NB_train_preds))
  # plot_confusion_matrix(gs_nb, X_train, y_train)
  # print("Accuracy with NB model's Train Data", end="")
  accuracy_NB_train = accuracy_score(y_train, NB_train_preds)
  # print(accuracy_NB_train)
  # print(classification_report(y_train, NB_train_preds, digits=3))
  f1_nb_train = f1_score(y_train, NB_train_preds, average='micro')
  # print("F1_Score Train SVM {}".format(f1_nb_train))

  NB_test_preds = gs_nb.predict(X_test)

  # print(confusion_matrix(test_target, NB_test_preds))
  # plot_confusion_matrix(gs_nb, X_test, y_test)
  # print("Accuracy with NB model's Test Data After Feature Selection", end="")
  accuracy_NB_test = accuracy_score(y_test, NB_test_preds)
  # print(accuracy_NB_test)
  # print(classification_report(y_test, NB_test_preds, digits=3))
  f1_nb_test = f1_score(y_test, NB_test_preds, average='micro')
  # print("F1_Score Train SVM {}".format(f1_nb_test))
  return accuracy_NB_train, accuracy_NB_test, f1_nb_train, f1_nb_test

#Method to experiment best number of features to extract from,
#using Select K Best method using chi2 for SVM Model
def experiment_Random_Forest(val, train, y):
  #Feature Selection with Select K Best and Chi2 Test
  X_new = SelectKBest(chi2, k=val).fit_transform(train, y)
  X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 42, stratify=y)
  
  gs_rf= GridSearchCV(RandomForestClassifier(),{
 'n_estimators': [200, 400, 800, 1400]}, cv=5, return_train_score=False)
  gs_rf.fit(X_train, y_train)
  gs_rf_df = pd.DataFrame(gs_rf.cv_results_)
  best_rf = gs_rf_df[gs_rf_df["mean_test_score"] == gs_rf_df["mean_test_score"].max()]
  rf_estimators_param = best_rf["param_n_estimators"]

  RandForest_train_preds = gs_rf.predict(X_train)

  # print(confusion_matrix(train_target, RandForest_train_preds))
  # plot_confusion_matrix(gs_rf, X_train, y_train)
  # print("Accuracy with RandForest model's Train Data", end="")
  accuracy_RandForest_train = accuracy_score(y_train, RandForest_train_preds)
  # print(accuracy_RandForest_train)
  # print(classification_report(y_train, RandForest_train_preds, digits=3))
  f1_dt_train = f1_score(y_train, RandForest_train_preds, average='micro')
  # print("F1_Score Train SVM {}".format(f1_dt_train))

  RandForest_test_preds = gs_rf.predict(X_test)

  # print(confusion_matrix(test_target, RandForest_test_preds))
  # plot_confusion_matrix(gs_rf, X_test, y_test)
  # print("Accuracy with RandForest model's Test Data After Feature Selection", end="")
  accuracy_RandForest_test = accuracy_score(y_test, RandForest_test_preds)
  # print(accuracy_RandForest_test)
  # print(classification_report(y_test, RandForest_test_preds, digits=3))
  f1_dt_test = f1_score(y_test, RandForest_test_preds, average='micro')
  # print("F1_Score Train SVM {}".format(f1_dt_test))
  return accuracy_RandForest_train, accuracy_RandForest_test, f1_dt_train, f1_dt_test

#Method to experiment best number of features to extract from,
#using Select K Best method using chi2 for Logisic Model
def experiment_Logistic(val, train, y):
  #Feature Selection with Select K Best and Chi2 Test
  X_new = SelectKBest(chi2, k=val).fit_transform(train, y)
  X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 42, stratify=y)
 
  gs_lr= GridSearchCV(LogisticRegression(),{
      'penalty': ["l2"],
      "solver": ["liblinear", "lbfgs"]
      }, cv=5, return_train_score=False)
  gs_lr.fit(X_train, y_train)
  gs_lr_df = pd.DataFrame(gs_lr.cv_results_)
  best_lr = gs_lr_df[gs_lr_df["mean_test_score"] == gs_lr_df["mean_test_score"].max()]
  lr_solver_param = best_lr["param_solver"]
  lr_penalty_param = best_lr["param_penalty"]
  LogReg_train_preds_fs = gs_lr.predict(X_train)
  accuracy_LogReg_train_fs = accuracy_score(y_train, LogReg_train_preds_fs)
  f1_LogReg_train_fs = f1_score(y_train, LogReg_train_preds_fs, average='micro')
  LogReg_test_preds_fs = gs_lr.predict(X_test)
  accuracy_LogReg_test_fs = accuracy_score(y_test, LogReg_test_preds_fs)
  f1_LogReg_test_fs = f1_score(y_test, LogReg_test_preds_fs, average='micro')
  return accuracy_LogReg_train_fs, accuracy_LogReg_test_fs, f1_LogReg_train_fs, f1_LogReg_test_fs

#Method to get experiment Best Model with multiple models,
#with hyperparameter tuning, with feature selection 
def getBestModel():
  train, y, tfidf = initialize()
  X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.20, random_state = 42, stratify=y)

  train_features = X_train
  test_features = X_test
  train_target = y_train
  test_target = y_test

  #Logistic Model Experimentation
  gs_lr= GridSearchCV(LogisticRegression(),{
      'penalty': ["l2"],
      "solver": ["liblinear", "lbfgs"]
      }, cv=5, return_train_score=False)
  gs_lr.fit(train_features, train_target)
  gs_lr_df = pd.DataFrame(gs_lr.cv_results_)
  best_lr = gs_lr_df[gs_lr_df["mean_test_score"] == gs_lr_df["mean_test_score"].max()]
  lr_solver_param = best_lr["param_solver"]
  lr_penalty_param = best_lr["param_penalty"]
  best_lr.head()

  #Testing the LogReg Model with best parameters obtained
  LogReg_train_preds = gs_lr.predict(train_features)

  plot_confusion_matrix(gs_lr, train_features, train_target)
  print("Accuracy with LogReg model's Train Data ", end="")
  accuracy_LogReg_train = accuracy_score(train_target, LogReg_train_preds)
  print(accuracy_LogReg_train)
  print(classification_report(train_target, LogReg_train_preds, digits=3))
  f1_LogReg_train = f1_score(train_target, LogReg_train_preds, average='micro')
  print("F1_Score Train LogReg {}".format(f1_LogReg_train))
  LogReg_test_preds = gs_lr.predict(test_features)
  plot_confusion_matrix(gs_lr, test_features, test_target)
  print("Accuracy with LogReg model's Test Data ", end="")
  accuracy_LogReg_test = accuracy_score(test_target, LogReg_test_preds)
  print(accuracy_LogReg_test)
  print(classification_report(test_target, LogReg_test_preds, digits=3))
  f1_LogReg_test = f1_score(test_target, LogReg_test_preds, average='micro')
  print("F1_Score Test LogReg {}".format(f1_LogReg_test))
  maxi_lg = 1
  max_ac_test_lg = 0
  train_acc_log_K = list()
  test_acc_log_K = list()
  train_f1_log_K = list()
  test_f1_log_K = list()
  for i in np.arange(1, 1500, 50):
    ac_train, ac_test, f1_train, f1_test = experiment_Logistic(i)
    train_acc_log_K.append(ac_train)
    test_acc_log_K.append(ac_test)
    train_f1_log_K.append(f1_train)
    test_f1_log_K.append(f1_test)
    if ac_test > max_ac_test_lg:
      maxi_lg = i
      max_ac_test_lg = ac_test
  print(maxi_lg, max_ac_test_lg)
  best_lr_ac_train, best_lr_ac_test, best_lr_f1_train, best_lr_f1_test = experiment_Logistic(maxi_lg)
  print("Best Metric Details After Feature Selection is")
  print("Train Accuracy {}".format(best_lr_ac_train))
  print("Test Accuracy {}".format(best_lr_ac_test))
  print("Train F1 Score {}".format(best_lr_f1_train))
  print("Test F1 Score {}".format(best_lr_f1_test))
  X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.20, random_state = 42, stratify=y)

  #Random forest Model Experimentation
  gs_rf= GridSearchCV(RandomForestClassifier(),{
  'n_estimators': [200, 400, 800, 1400]}, cv=5, return_train_score=False)
  gs_rf.fit(X_train, y_train)
  gs_rf_df = pd.DataFrame(gs_rf.cv_results_)
  best_rf = gs_rf_df[gs_rf_df["mean_test_score"] == gs_rf_df["mean_test_score"].max()]
  rf_estimators_param = best_rf["param_n_estimators"]

  #Testing the RandForest Model with best parameters obtained
  RandForest_train_preds = gs_rf.predict(X_train)

  plot_confusion_matrix(gs_rf, X_train, y_train)
  print("Accuracy with RandForest model's Train Data", end="")
  accuracy_RandForest_train = accuracy_score(y_train, RandForest_train_preds)
  print(accuracy_RandForest_train)
  print(classification_report(y_train, RandForest_train_preds, digits=3))
  f1_rf_train = f1_score(y_train, RandForest_train_preds, average='micro')
  print("F1_Score Train SVM {}".format(f1_rf_train))

  RandForest_test_preds = gs_rf.predict(X_test)
  plot_confusion_matrix(gs_rf, X_test, y_test)
  print("Accuracy with RandForest model's Test Data After Feature Selection", end="")
  accuracy_RandForest_test = accuracy_score(y_test, RandForest_test_preds)
  print(accuracy_RandForest_test)
  print(classification_report(y_test, RandForest_test_preds, digits=3))
  f1_rf_test = f1_score(y_test, RandForest_test_preds, average='micro')
  print("F1_Score Train SVM {}".format(f1_rf_test))

  maxi_rf = 1
  max_ac_test_rf = 0
  train_acc_rf_K = list()
  test_acc_rf_K = list()
  train_f1_rf_K = list()
  test_f1_rf_K = list()
  for i in np.arange(1, 1500, 400):
    ac_train, ac_test, f1_train, f1_test = experiment_Random_Forest(i)
    train_acc_rf_K.append(ac_train)
    test_acc_rf_K.append(ac_test)
    train_f1_rf_K.append(f1_train)
    test_f1_rf_K.append(f1_test)
    if ac_test > max_ac_test_rf:
      maxi_rf = i
      max_ac_test_rf = ac_test
  print(maxi_rf, max_ac_test_rf)
  best_rf_ac_train, best_rf_ac_test, best_rf_f1_train, best_rf_f1_test = experiment_Random_Forest(maxi_rf)
  print("Best Metric Details After Feature Selection is")
  print("Train Accuracy {}".format(best_rf_ac_train))
  print("Test Accuracy {}".format(best_rf_ac_test))
  print("Train F1 Score {}".format(best_rf_f1_train))
  print("Test F1 Score {}".format(best_rf_f1_test))

  #SVM Model Experimentation
  gs_svc = GridSearchCV(SVC(gamma='auto'), {
      'C': [1, 10, 20, 30, 40],
      'kernel': ['rbf','linear']
  }, cv=5, return_train_score=False)
  gs_svc.fit(X_train, y_train)
  gs_svc_df = pd.DataFrame(gs_svc.cv_results_)
  best_svc = gs_svc_df[gs_svc_df["mean_test_score"] == gs_svc_df["mean_test_score"].max()]
  svc_C_param = best_svc["param_C"]
  svc_kernel_param = best_svc["param_kernel"]

  #Testing the SVM Model with best parameters obtained
  SVM_train_preds = gs_svc.predict(X_train)
  plot_confusion_matrix(gs_svc, X_train, y_train)
  print("Accuracy with SVM model's Train Data ", end="")
  accuracy_SVM_train = accuracy_score(y_train, SVM_train_preds)
  print(accuracy_SVM_train)
  print(classification_report(y_train, SVM_train_preds, digits=3))
  f1_svm_train = f1_score(y_train, SVM_train_preds, average='micro')
  print("F1_Score Train SVM {}".format(f1_svm_train))
  SVM_test_preds = gs_svc.predict(X_test)
  plot_confusion_matrix(gs_svc, X_test, y_test)
  print("Accuracy with SVM model's Test Data ", end="")
  accuracy_SVM_test = accuracy_score(y_test, SVM_test_preds)
  print(accuracy_SVM_test)
  print(classification_report(y_test, SVM_test_preds, digits=3))
  f1_svm_test = f1_score(y_test, SVM_test_preds, average='micro')
  print("F1_Score Train SVM {}".format(f1_svm_test))
  maxi_svm = 1
  max_ac_test_svm = 0
  train_acc_svm_K = list()
  test_acc_svm_K = list()
  train_f1_svm_K = list()
  test_f1_svm_K = list()
  for i in np.arange(1, 1500, 100):
    ac_train, ac_test, f1_train, f1_test = experiment_SVM(i)
    train_acc_svm_K.append(ac_train)
    test_acc_svm_K.append(ac_test)
    train_f1_svm_K.append(f1_train)
    test_f1_svm_K.append(f1_test)
    if ac_test > max_ac_test_svm:
      maxi_svm = i
      max_ac_test_svm = ac_test
  print(maxi_svm, max_ac_test_svm)
  best_svm_ac_train, best_svm_ac_test, best_svm_f1_train, best_svm_f1_test = experiment_SVM(maxi_svm)
  print("Best Metric Details After Feature Selection is")
  print("Train Accuracy {}".format(best_svm_ac_train))
  print("Test Accuracy {}".format(best_svm_ac_test))
  print("Train F1 Score {}".format(best_svm_f1_train))
  print("Test F1 Score {}".format(best_svm_f1_test))


  #Bernoulli Bayes Model Experimentation
  params = {'alpha': [0.01, 0.1, 0.5, 1.0, 10.0],
            'fit_prior': [True, False],
            'binarize': [None, 0.0, 8.5, 10.0]
          }
  gs_nb = GridSearchCV(BernoulliNB(), param_grid=params, n_jobs=-1, cv=5, verbose=5)
  gs_nb.fit(X_train,y_train)
  gs_nb_df = pd.DataFrame(gs_nb.cv_results_)
  best_nb = gs_nb_df[gs_nb_df["mean_test_score"] == gs_nb_df["mean_test_score"].max()]

  #Testing the NB Model with best parameters obtained
  NB_train_preds = gs_nb.predict(X_train)
  plot_confusion_matrix(gs_nb, X_train, y_train)
  print("Accuracy with NB model's Train Data", end="")
  accuracy_NB_train = accuracy_score(y_train, NB_train_preds)
  print(accuracy_NB_train)
  print(classification_report(y_train, NB_train_preds, digits=3))
  f1_nb_train = f1_score(y_train, NB_train_preds, average='micro')
  print("F1_Score Train NB {}".format(f1_nb_train))
  NB_test_preds = gs_nb.predict(X_test)
  plot_confusion_matrix(gs_nb, X_test, y_test)
  print("Accuracy with NB model's Test Data After Feature Selection", end="")
  accuracy_NB_test = accuracy_score(y_test, NB_test_preds)
  print(accuracy_NB_test)
  print(classification_report(y_test, NB_test_preds, digits=3))
  f1_nb_test = f1_score(y_test, NB_test_preds, average='micro')
  print("F1_Score Train NB {}".format(f1_nb_test))
  maxi_nb = 1
  max_ac_test_nb = 0
  train_acc_nb_K = list()
  test_acc_nb_K = list()
  train_f1_nb_K = list()
  test_f1_nb_K = list()
  for i in np.arange(50, 1500, 50):
    ac_train, ac_test, f1_train, f1_test = experiment_NB(i)
    train_acc_nb_K.append(ac_train)
    test_acc_nb_K.append(ac_test)
    train_f1_nb_K.append(f1_train)
    test_f1_nb_K.append(f1_test)
    if ac_test > max_ac_test_nb:
      maxi_nb = i
      max_ac_test_nb = ac_test
  print(maxi_nb, max_ac_test_nb)
  best_nb_ac_train, best_nb_ac_test, best_nb_f1_train, best_nb_f1_test = experiment_NB(maxi_nb)
  print("Best Metric Details After Feature Selection is")
  print("Train Accuracy {}".format(best_nb_ac_train))
  print("Test Accuracy {}".format(best_nb_ac_test))
  print("Train F1 Score {}".format(best_nb_f1_train))
  print("Test F1 Score {}".format(best_nb_f1_test))

  #Plotting Train Test Accuracy Before Feature Selection 
  X = ['Logistic Regression','Randome Forest', 'SVM', 'Naives Bayes']
  X_axis = np.arange(len(X))
  train = [accuracy_LogReg_train, accuracy_RandForest_train, accuracy_SVM_train, accuracy_NB_train]
  test = [accuracy_LogReg_test, accuracy_RandForest_test, accuracy_SVM_test, accuracy_NB_test]
  plt.bar(X_axis - 0.2, train, 0.3, label = 'Train Accuracy')
  plt.bar(X_axis + 0.2, test, 0.3, label = 'Test Accuracy')
  plt.xticks(X_axis, X)
  plt.xlabel("Models")
  plt.ylabel("Metrics")
  plt.title("Model Accuracy Visualization")
  plt.legend()
  plt.show()

  #Plotting Train Test Accuracy After Feature Selection 
  X = ['Logistic Regression','Randome Forest', 'SVM', 'Naives Bayes']
  X_axis = np.arange(len(X))
  train = [best_lr_ac_train, best_rf_ac_train, best_svm_ac_train, best_nb_ac_train]
  test = [best_lr_ac_test, best_rf_ac_test, best_svm_ac_test, best_nb_ac_test]
  plt.bar(X_axis - 0.2, train, 0.3, label = 'Train Accuracy')
  plt.bar(X_axis + 0.2, test, 0.3, label = 'Test Accuracy')
  plt.xticks(X_axis, X)
  plt.xlabel("Models")
  plt.ylabel("Metrics")
  plt.title("Model Accuracy Visualization")
  plt.legend()
  plt.show()

  #Plotting Train Test F1_Scores Before Selection 
  X = ['Logistic Regression','Randome Forest', 'SVM', 'Naives Bayes']
  X_axis = np.arange(len(X))
  train = [f1_LogReg_train, f1_rf_train, f1_svm_train, f1_nb_train]
  test = [f1_LogReg_test, f1_rf_test, f1_svm_test, f1_nb_test]
  plt.bar(X_axis - 0.2, train, 0.3, label = 'Train Accuracy')
  plt.bar(X_axis + 0.2, test, 0.3, label = 'Test Accuracy')
  plt.xticks(X_axis, X)
  plt.xlabel("Models")
  plt.ylabel("Metrics")
  plt.title("Model Accuracy Visualization")
  plt.legend()
  plt.show()

  #Plotting Train Test F1_Scores After Selection 
  X = ['Logistic Regression','Randome Forest', 'SVM', 'Naives Bayes']
  X_axis = np.arange(len(X))
  train = [best_lr_f1_train, best_nb_f1_train, best_svm_f1_train, best_nb_f1_train]
  test = [best_lr_f1_test, best_nb_f1_test, best_svm_f1_test, best_nb_f1_test]
  plt.bar(X_axis - 0.2, train, 0.3, label = 'Train Accuracy')
  plt.bar(X_axis + 0.2, test, 0.3, label = 'Test Accuracy')
  plt.xticks(X_axis, X)
  plt.xlabel("Models")
  plt.ylabel("Metrics")
  plt.title("Model Accuracy Visualization After Feature Selection")
  plt.legend()
  plt.show()

  #Plotting for number of features for each model
  plt.figure(figsize=(12, 8))
  arr = list(np.arange(0, 1500, 50))
  plt.plot(arr, train_acc_log_K, label="Train Accuracy")
  plt.plot(arr, test_acc_log_K, label="Test Accuracy")
  plt.xlabel("Logistic Regression")
  plt.ylabel("Features Extracted ")
  plt.title("Metrics for Number of Features extracted")
  plt.legend()
  plt.show()

  plt.figure(figsize=(12, 8))
  arr = list(np.arange(1, 1500, 400))
  plt.plot(arr, train_acc_rf_K, label="Train Accuracy")
  plt.plot(arr, test_acc_rf_K, label="Test Accuracy")
  plt.xlabel("Random Forest Model")
  plt.ylabel("Features Extracted ")
  plt.title("Metrics for Number of Features extracted")
  plt.legend()
  plt.show()

  plt.figure(figsize=(12, 8))
  arr = list(np.arange(1, 1500, 100))
  plt.plot(arr, train_acc_svm_K, label="Train Accuracy")
  plt.plot(arr, test_acc_svm_K, label="Test Accuracy")
  plt.xlabel("SVM Model")
  plt.ylabel("Features Extracted ")
  plt.title("Metrics for Number of Features extracted")
  plt.legend()
  plt.show()

  plt.figure(figsize=(12, 8))
  arr = list(np.arange(50, 1500, 50))
  plt.plot(arr, train_acc_nb_K, label="Train Accuracy")
  plt.plot(arr, test_acc_nb_K, label="Test Accuracy")
  plt.xlabel("Naives Bayes Bernaulis")
  plt.ylabel("Features Extracted ")
  plt.title("Metrics for Number of Features extracted")
  plt.legend()
  plt.show()
