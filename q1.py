'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import cross_val_score
from collections import Counter
from sklearn.svm import SVC
import pandas as pandas

def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer(stop_words="english",min_df=0.001,ngram_range=(1,1))
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print(bow_train[:,bow_vectorize.vocabulary_.get('ax')])
    print(train_data.filenames[144])
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]),bow_train.sum(axis=0).argmax() )
    
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(stop_words='english',ngram_range=(1,1),min_df=0.001,binary=True, lowercase=False)
 
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
 #   print(np.array(tf_idf_train)[1:5])
    tf_idf_vectorize_test = TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def tf_cutoff(bow_train, tf_train, train_labels, cutoff=0):
    # training the baseline model
    mean_tf_idf = np.where(tf_train.mean(axis=0)>cutoff)[0]
   
    print("mean idf", tf_train.shape)
    bow_train = bow_train[:,mean_tf_idf]
  
    binary_train = (bow_train>0).astype(int)
  
    
    model = BernoulliNB()
    model.fit(binary_train, train_labels)
    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    return (train_pred == train_labels).mean()

#feautre selection based on variance
def feature_selection(train,test,thresh=0.00001):
    sel = VarianceThreshold(threshold=thresh)
    train= sel.fit_transform(train)
    test = sel.transform(test)
    return(train,test)

#mean tf idf cutoff for feature selection through cross validation --- didn't work
def cutoff_cv(bow_train,train_tf,train_labels, cutoff_min, cutoff_max,step,folds=5):
    split_cv = KFold(n_splits=folds)
    cutoffs = np.arange(cutoff_min,cutoff_max,step=step)
    accuracy = np.zeros((cutoffs.shape[0],folds))
    i=0
    for train_index,test_index in split_cv.split(bow_train):
        x_f_train, x_f_test = bow_train[train_index], bow_train[test_index]
        tf_train, tf_test = train_tf[train_index,:],train_tf[test_index,:]
        y_f_train, y_f_test = train_labels[train_index], train_labels[test_index]
        j=0
        for cutoff in cutoffs:
         accuracy[j,i] = tf_cutoff(x_f_train, tf_train,y_f_train,cutoff)
         j+=1
        i+=1
    print(accuracy)
        
def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    #evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return test_pred

def mnb(bow_train, train_labels):
    # multinomial naive bayes
    model = MultinomialNB()
    model.fit(bow_train, train_labels)


    train_pred = model.predict(bow_train)
    print('MultinomialNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    return model

def gnb(train, train_labels):
    # training the baseline model
    model = GaussianNB()
    model.fit(train.toarray(), train_labels)

    #evaluate the baseline model
    train_pred = model.predict(bow_train)
    print('GaussianNB train accuracy = {}'.format((train_pred == train_labels).mean()))
    return model

def bagging(train,train_labels):
    model = MultinomialNB()
    
    bag = BaggingClassifier(model,n_estimators=35,max_features=0.8,oob_score=True)
    model_grid = GridSearchCV(bag,param_grid=dict(max_features=np.arange(0.1,1,step=0.05)))
    model_grid.fit(train,train_labels)
    print(model_grid.best_params_)
    best_param = model_grid.best_params_['max_features']
    bag_best = BaggingClassifier(model,n_estimators=30,max_features=best_param,oob_score=True)
    bag_best.fit(train,train_labels)
    
    print("oob score: ", bag_best.oob_score_)
    pred = bag_best.predict(train)
    print("MNB + Bagging train accuracy", (pred == train_labels).mean())
    return(bag_best)

#random forest
def rfc(tf_train,tf_labels,tf_test, test_labels,C=1):
    model = sklearn.ensemble.RandomForestClassifier()
    model_grid = GridSearchCV(model,param_grid=dict(n_estimators=np.arange(7,21,step=1),max_features=np.arange(0,sqrt(tf_train.shape[1])/tf_train.shape[1],0.05),min_samples_leaf=np.arange(1,50,step=5)))
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    return(model_grid)

#svm
def svm_model(tf_train,tf_labels):
    model = SVC(decision_function_shape='ovo')
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(0.01,1,step=0.2),kernel = ['linear','rbf','poly','sigmoid']))
    nested_score = cross_val_score(model_grid,X=tf_train, y=tf_labels,cv = 5)
    print(nested_score)
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print("SVM training accuracy:", accuracy)
    return(model_grid)



def log_reg2(tf_train,tf_labels):
    model = sklearn.linear_model.LogisticRegression(penalty='l2',
                                                   solver='sag',
                                                   C=1,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='multinomial'
                                                   )
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(1,5,step=1)))
    nested_score = cross_val_score(model_grid,X=tf_train, y=tf_labels,cv = 5)
    print(nested_score)
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print("L2 logistic regression train accuracy: ", accuracy)
    return(model_grid, nested_score)


def log_reg(tf_train,tf_labels, tf_test, test_labels,C=1,):
    model = sklearn.linear_model.LogisticRegression(penalty='l1',
                                                   solver='liblinear',
                                                   C=C,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='ovr'
                                                   )
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(1,5,step=0.1)),cv=10)
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    best_c = model_grid.best_params_['C']
    model = sklearn.linear_model.LogisticRegression(penalty='l1',
                                                   solver='liblinear',
                                                   C=best_c,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='ovr'
                                                   )
    model.fit(tf_train,tf_labels)
    print(model.coef_.shape)
    print(len(model.coef_ >0))
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    return(model)

def run_test(model,data,truth,model_name):
    pred = model.predict(data)
    accuracy = (pred == truth).mean()
    print(model_name, " test accuracy: ", accuracy)
    return(pred)

def confusion_matrix(pred,true, target_names,filename):
    cm = np.zeros((20,20))
    pred_truth = np.zeros(pred.shape[0],dtype={'names':('predicted','truth'),'formats':('int8','int8')})
    pred_truth['predicted'] = pred
    pred_truth['truth']= true
    for i in np.arange(0,20,step=1):
        for j in np.arange(0,20,step=1):
            cm[i,j] = pred_truth[(pred_truth['predicted'] == i ) & (pred_truth['truth']==j)].shape[0]
    df = pandas.DataFrame(data=cm,index=target_names)
   
    df.columns = target_names
    df.to_csv(filename,float_format='% 6.0f')
    print('total entries: ',cm.sum())
    return(cm)

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    
    train_tf, test_tf, feature_names = tf_idf_features(train_data,test_data)
    train_tf,test_tf = feature_selection(train_tf,test_tf,0.00001)
    #tf_train.min(axis=1)
   # log_reg(train_tf,train_data.target,test_tf, test_data.target)
  #  log_reg2(train_bow,train_data.target,test_bow, test_data.target)
   # cv_cutoff(train_bow,train_labels,train_tf,)
 #   bnb_cutoff_model = cutoff_cv(train_bow, train_tf, train_data.target,cutoff_min=0, cutoff_max = (train_tf.mean(axis=0)).max(axis=1), step = ((train_tf.mean(axis=0)).max(axis=1))/10 )
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
   
    mnb_model = mnb(train_bow, train_data.target)
    mnb_pred = run_test(mnb_model,test_bow,test_data.target,"MultinomialNB")
    confusion_matrix(mnb_pred,test_data.target,test_data.target_names,'mnb_confusion.csv')
    
    bag_model = bagging(train_bow, train_data.target)
    bag_pred = run_test(bag_model,test_bow,test_data.target,"Bagged MultinomialNB")
    confusion_matrix(bag_pred,test_data.target,test_data.target_names,'bg_mnb_confusion.csv')
    
    log_reg_model, lg_nest = log_reg2(train_tf,train_data.target)
    log_pred = run_test(log_reg_model,test_tf,test_data.target,"L2 penalized logistic regression")
    confusion_matrix(log_pred,test_data.target,test_data.target_names,'log_confusion.csv')
    
    svm_out = svm_model(train_tf, train_data.target)
    svm_pred = run_test(svm_out,test_tf,test_data.target,"SVM")
    confusion_matrix(svm_pred,test_data.target,test_data.target_names,'svm_confusion.csv')                    
 
    rfc_mod = rfc(train_tf,train_data.target,test_tf, test_data.target)
    rfc_pred = run_test(rfc_mod,test_tf,test_data.target,"Random Forest")
    confusion_matrix(rfc_pred,test_data.target,test_data.target_names,'rfc_confusion.csv') 