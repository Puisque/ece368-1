import numpy as np
import matplotlib.pyplot as plt
import util

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    # Determine parameters of the prior distribution, p(a) ~ N(mu_a, cov_a)
    mu_a = np.array([[0], [0]])
    cov_a = np.array([[beta, 0], [0, beta]])
    
    # Plot the contours of the prior distribution
    x_axis = np.arange(-1, 1, 0.01)
    y_axis = np.arange(-1, 1, 0.01)
    
    X_label, Y_label = np.meshgrid(x_axis, y_axis)    
    x_set = np.dstack((X_label, Y_label))
    x_set = np.reshape(x_set, (len(X_label)*len(Y_label), 2))

    p_of_a = util.density_Gaussian(mu_a.transpose(), cov_a, x_set).reshape(X_label.shape[0], X_label.shape[1])
    
    # Clean up graph
    plot_one = plt.figure(1)
    plot_one_axes = plot_one.add_axes([0.1, 0.1, 0.8, 0.8])
    
    plt.contour(X_label, Y_label, p_of_a)
    a_val = plt.scatter(-0.1, -0.5, color = 'blue', s = 50)
    
    plot_one_axes.set_xlim([-1, 1])
    plot_one_axes.set_ylim([-1, 1])
    
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('Prior Distribution p(a)')  
    plt.legend([a_val],
               ['(a0, a1)'],
               scatterpoints = 1,
               loc = 'best')
    
    # plt.savefig("prior.pdf")
    return 
    
def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    ### TODO: Write your code here
       
    # Intermediate matrices required for calculations
    size_of_x = np.size(x,0)
    temp_ones = np.ones((1,size_of_x), dtype=int)
    x_transpose = np.vstack((temp_ones, np.transpose(x)))
    
    id_matrix = (1 / beta) * np.identity(2)
    
    # Determine parameters of the posterior distribution, p(a|x,z) ~ N(mu, Cov)
    Cov = np.linalg.inv(id_matrix + (1 / sigma2) * np.matmul(x_transpose, np.transpose(x_transpose)))    
    mu = (1 / sigma2) * np.matmul(np.matmul(Cov, x_transpose), z)
       
    # Plot the contours of the posterior distribution
    x_axis = np.arange(-1, 1, 0.01)
    y_axis = np.arange(-1, 1, 0.01)
    
    X_label, Y_label = np.meshgrid(x_axis, y_axis)    
    x_set = np.dstack((X_label, Y_label))
    x_set = np.reshape(x_set, (len(X_label)*len(Y_label), 2))

    p_of_a_given_z = util.density_Gaussian(mu.transpose(), Cov, x_set).reshape(X_label.shape[0], X_label.shape[1])
    
    # Clean up graph
    plot_two = plt.figure(2)
    plot_two_axes = plot_two.add_axes([0.1, 0.1, 0.8, 0.8])
    
    plt.contour(X_label, Y_label, p_of_a_given_z)
    a_val = plt.scatter(-0.1, -0.5, color = 'blue', s = 50)

    plot_two_axes.set_xlim([-1, 1])
    plot_two_axes.set_ylim([-1, 1])
    
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('Posterior Distribution p(a|x,z) for N = ' + str(size_of_x))  
    plt.legend([a_val],
               ['(a0, a1)'],
               scatterpoints = 1,
               loc = 'best')
    
    # plt.savefig("posterior" + str(size_of_x) + ".pdf")

    return (mu,Cov)

def predictionDistribution(x,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    ### TODO: Write your code here
    
    size_of_x = np.size(x_train, 0)

    plot_three = plt.figure(3)
    plot_three_axes = plot_three.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Plot the new inputs and predicted targets as well as the standard deviation
    for i in x:    
        x_mat = np.array([[1], [i]])
        
        # Determine parameters of the likelihood distribution, p(z|a,x,z) ~ N(mu, Cov)
        mu_z = np.matmul(np.transpose(mu), x_mat)[0][0]
        cov_z = np.matmul(np.transpose(x_mat), np.matmul(Cov, x_mat))[0][0]
        std_dev = np.sqrt(cov_z)
        
        req1and2 = plt.errorbar(i, mu_z, yerr = std_dev, fmt = '.', color = 'grey', 
                                ecolor = 'lightgrey', elinewidth = 2, capsize = 3)

    # Plot the training samples used in the calculation of the posterior distribution
    for j in range(0, len(x_train)):
        req3 = plt.scatter(x_train[j][0], z_train[j][0], color = 'red', s = 7)
        
    # Clean up graph
    plot_three_axes.set_xlim([-4, 4])
    plot_three_axes.set_ylim([-4, 4])
    
    plt.xlabel('Input (x)')
    plt.ylabel('Target (z)')
    plt.title('Prediction for New Data Set with N = ' + str(size_of_x) + ' Training Samples')  
    plt.legend([req1and2, req3],
               ['New Data', 'Training Samples'],
               scatterpoints = 1,
               loc = 'best')
    
    # plt.savefig("predict" + str(size_of_x) + ".pdf")
    return 

if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 1
    
    # number of training samples used to compute posterior
    ns  = 1
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x,z,beta,sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    
    print("Done")
   