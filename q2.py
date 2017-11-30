import numpy as np 

from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt

np.random.seed(1847)

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch  

class GDOptimizer(object):
    '''
    A gradient descent optimizer with momentum
    '''

    def __init__(self, lr, param_shape, beta=0.0):
        self.lr = lr
        self.beta = beta
        self.iter = 0
        self.update_old = np.zeros(param_shape)
        

    def update_params(self, params, grad):
        # Update parameters using GD with momentum and return
        # the updated parameters
        params_old = params
        grad_old = grad(params_old)
        update = -self.lr*grad_old+self.beta*self.update_old
        params = params + update
        self.update_old = update
        return params

class SVM(object):
    '''
    A Support Vector Machine
    '''

    def __init__(self, c, feature_count):
        self.c = c
        self.w = np.random.normal(0.0, 0.1, feature_count)
        
        
    def hinge_loss(self, X, y):
        '''
        Compute the hinge-loss for input data X (shape (n, m)) with target y (shape (n,)).

        Returns a length-n vector containing the hinge-loss per data point.
        '''
        # Implement hinge loss
        return None
    def train(self,X,y,iters,batchsize,optimizer,C):
        for i in np.arange(0,iters):
            sample = BatchSampler(X,y,batchsize)
            samp_X, samp_y = sample.get_batch()
            w = self.w
            y_x_sum = np.zeros((w.shape[0],1))
            for s in np.arange(0,batchsize):
                y_s = samp_y[s]
                x_s = samp_X[s,:]
                y_fx = np.dot(y_s,np.dot(w,x_s))
                if y_fx < 1:
                     add_row = np.array(y_s*x_s)
                     add_row = np.reshape(add_row,(add_row.shape[0],1))
                     y_x_sum = np.append(y_x_sum,add_row,axis=1)
            print(y_x_sum.shape)
            y_x_sum = np.sum(y_x_sum,axis=1)
            print(y_x_sum.shape)
            def grad(w):
                return(w+C/batchsize*y_x_sum)
            update_grad = optimizer.update_params(w,grad)
                            
                
                
                        
        
         #   print(dist_to_marg)
    def grad(self, X, y):
        '''
        Compute the gradient of the SVM objective for input data X (shape (n, m))
        with target y (shape (n,))

        Returns the gradient with respect to the SVM parameters (shape (m,)).
        '''
        # Compute (sub-)gradient of SVM objective
        return None

    def classify(self, X):
        '''
        Classify new input data matrix (shape (n,m)).

        Returns the predicted class labels (shape (n,))
        '''
        # Classify points as +1 or -1
        return None

def load_data():
    '''
    Load MNIST data (4 and 9 only) and split into train and test
    '''
    mnist = fetch_mldata('MNIST original', data_home='./data')
    label_4 = (mnist.target == 4)
    label_9 = (mnist.target == 9)

    data_4, targets_4 = mnist.data[label_4], np.ones(np.sum(label_4))
    data_9, targets_9 = mnist.data[label_9], -np.ones(np.sum(label_9))

    data = np.concatenate([data_4, data_9], 0)
    data = data / 255.0
    targets = np.concatenate([targets_4, targets_9], 0)

    permuted = np.random.permutation(data.shape[0])
    train_size = int(np.floor(data.shape[0] * 0.8))

    train_data, train_targets = data[permuted[:train_size]], targets[permuted[:train_size]]
    test_data, test_targets = data[permuted[train_size:]], targets[permuted[train_size:]]
    print("Data Loaded")
    print("Train size: {}".format(train_size))
    print("Test size: {}".format(data.shape[0] - train_size))
    print("-------------------------------")
    return train_data, train_targets, test_data, test_targets

def optimize_test_function(optimizer, w_init=10.0, steps=300):
    '''
    Optimize the simple quadratic test function and return the parameter history.
    '''
    def func(x):
        return 0.01 * x * x

    def func_grad(x):
        return 0.02 * x

    w = w_init
    w_history = [w_init]
    print(w_history)
    

    for x in range(steps):
        # Optimize and update the history
        params_w = grad.update_params(w,func_grad)
        print(params_w)
        w_history.append(params_w)
        w = params_w
        pass
    return w_history

def optimize_svm(train_data, train_targets, penalty, optimizer, batchsize, iters):
    '''
    Optimize the SVM with the given hyperparameters. Return the trained SVM.
    '''
    print(train_data.shape)
    train_data = np.append(np.ones((train_data.shape[0],1)),train_data,axis=0)
    print(train_data.shape)
    svm = SVM(penalty, feature_count=train_data.shape[1])
    svm.train(train_data,train_targets,iters,batchsize,optimizer,penalty)
   # svm.hinge(train_data,train_targets)
    return None

if __name__ == '__main__':
    w = np.array(10.0)
    grad = GDOptimizer(lr =10,beta=0,param_shape = w.shape)
   
 #   w_hist = optimize_test_function(grad,w_init=w)
    train_data, train_targets, test_data, test_targets = load_data()
    svm_opt = GDOptimizer(lr=0.05,beta=0, param_shape = train_data.shape[1])
    my_svm = optimize_svm(train_data, train_targets, penalty =1, optimizer=svm_opt, batchsize=10, iters=2)
    pass
