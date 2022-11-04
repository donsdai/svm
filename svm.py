import os
import click
import pandas as pd
import time
from sklearn import svm, metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 10000 data -> 300 epoch
# 50000 data -> 1000 epoch
@click.command()
@click.option('--filename', default='data-11757.csv', help='Data source')
@click.option('--dr', default=None, help='Number of data to be processed.')
@click.option('--lr', default=0.01, help='Learning rate. Default 0.01')
def svm(filename, dr, lr):
  # read file from filename parameter
  filepath = (os.path.join('./dataset/new/', filename))
  if dr is None:
    df_cc = pd.read_csv(filepath, delimiter=',')    
  else:
    df_cc = pd.read_csv(filepath, delimiter=',', nrows = int(dr)) 

  # get fitur data 
  X = df_cc.drop(columns=['Time', 'Amount'])  
  # get label data
  y = df_cc['Class']

  # random state using timestamp so it will be different everytime
  epoch_time = int(time.time())
  
  # Split data to training and test data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=epoch_time)                                          
  
  # train data with Linear SVC
  clf = LinearSVC(random_state=epoch_time, tol=lr)
  clf.fit(X_train, y_train.ravel())
  y_pred = clf.predict(X_test)

  # processing and print confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  print(cm)
  
  # show other informations
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)
  print("Accuracy:", accuracy)
  print("Precision:", precision)
  print("Recall:", recall)

if __name__ == '__main__':
  svm()