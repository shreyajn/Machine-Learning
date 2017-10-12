__title__ = "HW1 Q6"
__author__ = "Wonkyung Kim(wk2294), Shreya Jain(sj2842), Yu Bai(yb2300)"
__date__ = "$Oct 8, 2017"

import sys
import numpy as np
import scipy.io as spio
import math
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxint)


def one_hot_encode( label_list ):
    """
    Get one-hot-encoding vectors from the label.

    """
    encoded_list = []

    for i in range( len( label_list ) ):
        encoded_num = []
        for j in range( 10 ):
            if j == label_list[i,0]:
                encoded_num.append( 1 )
            else:
                encoded_num.append( 0 )
        encoded_list.append( encoded_num )

    return np.matrix( encoded_list )


def op_func( theta, X, Y ):
    """
    Get optimal function value.

    """

    result = 0
    for i in range( X.shape[0] ):
        x_i = X[i]
        y_i = Y[i]
        theta_t = theta.transpose()

        mat = -2 * np.dot( theta_t, x_i )
        mat += np.multiply( x_i, x_i ) + np.multiply( theta, theta ).transpose()
        mat *= 1/2.0
        mat = np.dot( y_i, mat)
    
        result += mat
    
    return np.sum( result )


########################################################################
# Code which follows the exact same step as homework description       #
# Too slow to calculate this large data                                #
# op_func() is the Matrix/vectorized version of this                   #
########################################################################
def op_func2( theta, X, Y ):
    result = 0
    for i in range( X.shape[0] ):
        for k in range( Y.shape[1] ):
            for d in range( X.shape[1] ):
                cal = 0.5 * Y[i, k] * ( ( X[i, d] - theta[0, k] ) ** 2 )
                result += cal

    return result


# Question (i)
def get_grad_f( theta, X, Y  ):
    """
    Get gradient f value.

    """
    result = []
    for k in range( label.shape[1] ):
        y_k = Y[:, k]
        theta_k = theta[0, k]
        cal = y_k.transpose()
        cal = np.dot( cal, ( -1 * X ) + theta_k )
        cal = np.sum( cal )
        result.append( cal )

    result = np.matrix( result )

    return result

########################################################################
# Code of the gradient value                                           #
# Too slow to calculate this large data                                #
# get_grad_f() is the Matrix/vectorized version of this                #
########################################################################
def get_grad_f2( theta, X, Y  ):
    """
    Get gradient f value.

    """
    result = []
    for k in range( label.shape[1] ):
        sum = 0
        for i in range( X.shape[0] ):
            for d in range( X.shape[1] ):
                cal = Y[i, k] * ( -1 * X[i, d] + theta[0,k] )
                sum += cal
        print sum
        result.append( sum )

    result = np.matrix( result )

    return result



# Question (iii)
def get_grad_f_i( theta, X, Y, i ):
    """
    Get gradient f_i value for i-1th data

    """
    result = []
    for k in range( label.shape[1] ):
        y_k = Y[i, k]
        theta_k = theta[0, k]
        cal = y_k.transpose()
        cal = np.dot( cal, ( -1 * X[i,:] ) + theta_k )
        cal = np.sum( cal )
        result.append( cal )
    result = np.matrix( result )

    return result


def get_all_grad_f_i( theta, X, Y ):
    """
    Get all grad_f_i and store into a liast
    Output the list

    """
    num_of_data = X.shape[0]
    
    grads = [] # Matrix of all grad_f_i
    for i in range( num_of_data ):
        grad_f_i = get_grad_f_i( theta, X, Y, i )
        grads.append( grad_f_i )

    return grads

   


# Question (ii)
def get_op_theta( theta, X, Y ):
    """
    Get optimazed theta which minimizes the function f
    Output optimized theta value, 
    the list of time wall: x-axis,
    and the list of the value: y-axis 
    """

    # Get difference of uclidean distance
    def get_difference( old_theta, new_theta ):
        difference_mat = old_theta - new_theta
        difference_square = np.multiply( difference_mat, difference_mat )
        difference = math.sqrt( np.sum( difference_square ) )
        
        return difference

    # Get updated theta
    def get_new_theta( old_theta, eta ):
        grad_val = get_grad_f( old_theta, X, Y )
        new_theta = old_theta - ( eta * grad_val )
        return new_theta

    ############################################################
    precision = 0.01                                           #
    eta = 0.000000008                                         #
    time_list = []                                             #
    value_list = []                                            #
    ############################################################
    
    old_theta = theta
    new_theta = get_new_theta( old_theta, eta )
    
    difference = get_difference( old_theta, new_theta )

    while difference > precision:
        old_theta = new_theta
        new_theta = get_new_theta( old_theta, eta )
        # Get new difference
        difference = get_difference( old_theta, new_theta )

        
        # Update time_list and value_list to make a plot
        cur_time = time.clock()
        time_list.append( cur_time )
        value = op_func( new_theta, X, Y )
        value_list.append( value )
        
        # Showing Information...
        print
        print "difference: " + str( difference )
        print "theta: "
        print new_theta
        print "function value: " + str( value )

    return new_theta, time_list, value_list




# Question (iv)
def get_op_theta_fast( theta, X, Y ):
    """
    Get optimazed theta which minimizes the function f
    Output optimized theta value, 
    the list of time wall: x-axis,
    and the list of the value: y-axis 
    """
    # Get difference of uclidean distance
    def get_difference( old_theta, new_theta ):
        difference_mat = old_theta - new_theta
        difference_square = np.multiply( difference_mat, difference_mat )
        difference = math.sqrt( np.sum( difference_square ) )
        
        return difference

    # Mini_batch example!!
    def get_mini_batch_grad( theta ):
        random.seed( 1000 ) 
        grad_sum = None
        size = 256

        for i in range( size ):
            random.seed()
            rand_num = random.randint( 0, X.shape[0] - 1 )
            grad = get_grad_f_i( theta, X, Y, rand_num )
            if grad_sum == None:
                grad_sum = grad
            else:
                grad_sum = grad_sum + grad

        return grad_sum / size
    
    # Set random seed
    random.seed( 1 )
    
    # Get updated theta
    def get_new_theta( old_theta, eta ):

        # Code for using single sample gradient
        random_i = random.randint( 0, X.shape[0] - 1 )
        grad_val = get_grad_f_i( old_theta, X, Y, random_i )
        # Scale by the size N (multiply by 10,000)
        grad_val = grad_val * X.shape[0]
        
        new_theta = old_theta - ( eta * grad_val )

        '''Code for Using Mini-batch'''
        #grad_val = get_mini_batch_grad( old_theta )
        #grad_val = grad_val * X.shape[0]
        #new_theta = old_theta - ( eta * grad_val )

        return new_theta

    ############################################################
    precision = 0.01                                           #
    eta = 0.000000008                                         #
    time_list = []                                             #
    value_list = []                                            #
    ############################################################

    old_theta = theta
    new_theta = get_new_theta( old_theta, eta )

    difference = get_difference( old_theta, new_theta )

    while difference > precision:
        old_theta = new_theta
        new_theta = get_new_theta( old_theta, eta )
        # Get new difference
        difference = get_difference( old_theta, new_theta )

        # Update time_list and value_list to make a plot
        cur_time = time.clock()
        time_list.append( cur_time )
        value = op_func( new_theta, X, Y )
        value_list.append( value )
        
        # Showing Information...
        print
        print "difference: " + str( difference )
        print "theta: "
        print new_theta
        print "function value: " + str( value )

    #return new_theta, grad_val_observe, time_list, value_list
    return new_theta, time_list, value_list









# Question (v)
def get_all_gradients_for_Q4( theta, X, Y ):
    """
    Do the same thing as Q(iv) but it is actually only for storing and
    observing the sample gradient and whole gradient for the Q(iv) step
    Output the sample grdient and whole grdient data

    """
    # Get difference of uclidean distance
    def get_difference( old_theta, new_theta ):
        difference_mat = old_theta - new_theta
        difference_square = np.multiply( difference_mat, difference_mat )
        difference = math.sqrt( np.sum( difference_square ) )
        
        return difference

    # Contains all gradient_i
    grad_i_val_observe = []
    grad_val_observe = []

    # Set random seed
    random.seed( 1 )
    
    # Get updated theta
    def get_new_theta( old_theta, eta ):

        # Code for using single sample gradient
        random_i = random.randint( 0, X.shape[0] - 1 )
        grad_i_val = get_grad_f_i( old_theta, X, Y, random_i )
        # Get the whole gradient to observe
        grad_val = get_grad_f( old_theta, X, Y )

        # Scale by the size N (multiply by 10,000)
        grad_i_val = grad_i_val * X.shape[0]
        
        # Store grad_val to observe Q(v)
        grad_i_val_list = grad_i_val.tolist()
        grad_i_val_list = grad_i_val_list[0]
        grad_val_list = grad_val.tolist()
        grad_val_list = grad_val_list[0]

        grad_i_val_observe.append( grad_i_val_list )
        grad_val_observe.append( grad_val_list )
        new_theta = old_theta - ( eta * grad_i_val )
        
        return new_theta

    ############################################################
    precision = 0.01                                           #
    eta = 0.000000008                                         #
    ############################################################

    old_theta = theta
    new_theta = get_new_theta( old_theta, eta )

    difference = get_difference( old_theta, new_theta )

    while difference > precision:
        old_theta = new_theta
        new_theta = get_new_theta( old_theta, eta )
        # Get new difference
        difference = get_difference( old_theta, new_theta )
        
        value = op_func( new_theta, X, Y )
        
        # Showing information...
        print
        print "difference: " + str( difference )
        print "theta: "
        print new_theta
        print "function value: " + str( value )

    return grad_i_val_observe, grad_val_observe








if __name__ == "__main__":

    mat = spio.loadmat( "hw1data.mat" )
    data = np.matrix( mat["X"] )
    label = mat["Y"]
    label = one_hot_encode( label )
    # arbitrary starting theta
    theta = np.matrix( [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] )

    # Make sure double precision
    data = data.astype( float )
    label = label.astype( float )
    theta = theta.astype( float )


    
    # Code for Q2
    '''
    op, time, value = get_op_theta( theta, data, label )
    print op
    print "time"
    print time
    print "value"
    print value
    print
     
    plt.plot( time, value )
    plt.title("Q2" ) 
    plt.xlabel( 'Time' )
    plt.ylabel( 'Value' )
    plt.show()
    '''

    # Result of Q2
    #difference: 0.00994694761547
    #theta:
    #[[ 44.57708335  19.8437914   37.72223372  36.06837667  30.94106278
    #   31.78815657  34.71347163  29.14377477  37.74894346  31.4059688 ]]
    #function value: 24298968782.1

    
    # Code for Q4
    '''
    op_theta, time, value = get_op_theta_fast( theta, data, label )
    print
    print "time"
    print time
    print
    print "value"
    print value
    plt.plot( time, value )
    plt.title("Q6" ) 
    plt.xlabel( 'Time' )
    plt.ylabel( 'Value' )
    plt.show()
    '''

    # Result of Q4
    # difference: 0.00836123518975
    # theta:
    #    [[ 44.27442517  18.92473978  34.2421971   35.53165227  29.22544425
    #       29.75770379  35.12970592  30.83726659  34.03855463  30.05652166]]
    # function value: 24318995689.0


    # Code for Q5
    '''
    # Making 10 histograms for gradient and sample gradients data
    grad_i, grad = get_all_gradients_for_Q4( theta, data, label )
    # For each k, make the histogram
    for k in range( len( grad[0] ) ):
        grad_i_k = []
        grad_k = []
        
        for list in grad_i:
            grad_i_k.append( list[k] )
        for list in grad:
            grad_k.append( list[k] )

        plt.hist( grad_i_k, label='sampe gradients k' )
        plt.hist( grad_k, label='gradient k' )
        plt.title( "k = " + str(k) )
        plt.xlabel( 'Value' )
        plt.ylabel( 'Frequent' )
        plt.legend( loc='upper right' )
        plt.show()
    '''
