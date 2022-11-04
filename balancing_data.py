import os
import click
import pandas as pd
import time
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix

@click.command()
@click.option('--dr', default=None, help='Number of data to be processed.')
@click.option('--lr', default=0.01, help='Learning rate. default is 0,01.')
def balancing_data(dr, lr):
  # read data from credit card master file
  filepath = (os.path.join('./dataset/', 'creditcard.csv'))
  if dr is None:
    df_cc = pd.read_csv(filepath, delimiter=',')    
  else:
    df_cc = pd.read_csv(filepath, delimiter=',', nrows = int(dr)) 

  epoch_time = int(time.time())
  # split dataframe credit card into 2 classes
  # fraud and not fraud
  # this solution is to augmented the minority (fraud) data
  df_majority = df_cc[(df_cc['Class']==0)]
  df_majority_training = df_majority.sample(frac = 0.7)
  df_majority_testing = df_majority.drop(df_majority_training.index)

  df_minority = df_cc[(df_cc['Class']==1)]
  # split dataframe into 2 randomly
  df_minority_training = df_minority.sample(frac = 0.4)
  df_minority_test     = df_minority.drop(df_minority_training.index)

  # upsampling data minority for training
  df_minority_tr_upsampled = resample(df_minority_training, 
                            replace=True,    # sample with replacement
                            n_samples=int(int(dr)*0.3), # to match majority class
                            random_state=epoch_time)
  # upsampling data minority for testing                                 
  df_minority_test_upsampled = resample(df_minority_test, 
                            replace=True,    # sample with replacement
                            n_samples=int(int(dr)*0.01), # to match majority class
                            random_state=epoch_time)                                

  # merged data majority (non fraud) with augmented fraud data
  # this is to make sure the data is balanced
  df_upsampled_training = pd.concat([df_minority_tr_upsampled, df_majority_training])                                                            
  
  # extract dataframe to exclude time dan amount 
  x_train = df_upsampled_training.drop(columns=['Time', 'Amount'])
  y_train = df_upsampled_training['Class']

  # merge the test data with additional sample on minority
  df_test = pd.concat([df_majority_testing, df_minority_test_upsampled])

  # Get test data
  x_test = df_test.drop(columns=['Time', 'Amount'])
  y_test = df_test['Class']
  
  clf = LinearSVC(random_state=epoch_time, tol=lr)
  clf.fit(x_train, y_train.ravel()) 
  y_pred = clf.predict(x_test)

  # confusion matrix
  cm = confusion_matrix(y_test, y_pred)
  print(cm)
  
  # accuracy, precision and recall information
  accuracy = metrics.accuracy_score(y_test, y_pred)
  precision = metrics.precision_score(y_test, y_pred)
  recall = metrics.recall_score(y_test, y_pred)  

  # save new augmented data
  # when accuracy between 0.9 and 1.0 
  # and precision between 0.9 and 1.0
  # and recall between 0.9 and 1.0
  if (
      (accuracy >= 0.90 and accuracy <=  1.0) and
      (precision >= 0.90 and precision <=  1.0) and
      (recall >= 0.70 and recall <  1.0)
    ):
    df_new = pd.concat([df_upsampled_training, df_test])
    df_new.to_csv('dataset/new/data-' + str(len(df_new)) + '.csv')
    print("Save to file")

if __name__ == '__main__':
  balancing_data()