'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(stop_words='english')
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

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

    return model

def rfc(tf_train,tf_labels,tf_test, test_labels,C=1):
    model = sklearn.ensemble.RandomForestClassifier(penalty='l2',
                                                   solver='sag',
                                                   C=C,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='multinomial'
                                                   )
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(0.01,1,step=0.1)))
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    test_pred = model_grid.predict(tf_test)
    accuracy = (test_pred == test_labels).mean()
    print(accuracy)
    pass
def log_reg2(tf_train,tf_labels,tf_test, test_labels,C=1):
    model = sklearn.linear_model.LogisticRegression(penalty='l2',
                                                   solver='sag',
                                                   C=C,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='multinomial'
                                                   )
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(0.01,1,step=0.1)))
    model_grid.fit(tf_train,tf_labels)
    print(model_grid.best_params_)
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    test_pred = model_grid.predict(tf_test)
    accuracy = (test_pred == test_labels).mean()
    print(accuracy)
    pass


def log_reg(tf_train,tf_labels, tf_test, test_labels,C=1,):
    model = sklearn.linear_model.LogisticRegression(penalty='l1',
                                                   solver='liblinear',
                                                   C=C,
                                                   fit_intercept = True,
                                                   random_state=42,
                                                   max_iter= 200,
                                                   multi_class ='ovr'
                                                   )
    model_grid = GridSearchCV(model,param_grid=dict(C=np.arange(0.01,1,step=0.1)))
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
    print(len(model.coef_[model.coef_ != 0]))
    train_pred = model_grid.predict(tf_train)
    accuracy = (train_pred == tf_labels).mean()
    print(accuracy)
    test_pred = model_grid.predict(tf_test)
    accuracy = (test_pred == test_labels).mean()
    print(accuracy)
    pass

if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)
    print(train_bow.shape)
    print(sklearn.__version__)
    train_tf, test_tf, feature_names = tf_idf_features(train_data,test_data)
    print(train_tf.shape)
    log_reg(train_tf,train_data.target,test_tf, test_data.target)
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    log_reg2(train_tf,train_data.target,test_tf, test_data.target)
