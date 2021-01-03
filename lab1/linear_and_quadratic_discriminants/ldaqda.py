import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here
    
    # define variables used in the ML estimator calculations    
    N_m = 0
    N_f = 0
    
    for i in y:
        if i == 1:
            N_m += 1
        elif i == 2:
            N_f += 1

    # separate the data according to male and female samples      
    male_data = np.zeros((N_m, 2))
    female_data = np.zeros((N_f, 2))
    
    count_male = 0
    count_female = 0
    for each_person in range(0, len(x)):
        if y[each_person] == 1:
            male_data[count_male] = x[each_person]
            count_male += 1
        elif y[each_person] == 2:
            female_data[count_female] = x[each_person]
            count_female += 1
    
    # ----------------------------------------------------------------------- #
    # calculate the MLE of required parameters
    # ----------------------------------------------------------------------- #
    
    # mean for male
    mu_male = [0, 0]
    for i in male_data:
        mu_male += i / N_m
        
    # mean for female
    mu_female = [0, 0]
    for i in female_data:
        mu_female += i / N_f    
    
    # covariance matrix for LDA
    mu = [0, 0]
    for i in x:
        mu += i / (N_m + N_f)
    
    cov = np.zeros((2,2))

    x_data = np.zeros((N_m + N_f, 2))
    for j in range(0, len(x)):
        temp_matrix = np.zeros((1,2))
        temp_matrix[0] = x[j] - mu
        
        cov = cov + np.matmul(temp_matrix.transpose(), temp_matrix)
    
    cov = (1 / (N_m + N_f)) * cov
    
    # covariance matrix for QDA (male)    
    cov_male = np.zeros((2,2))
    
    for j in range(0, len(male_data)):
        temp_matrix = np.zeros((1,2))
        temp_matrix[0] = male_data[j] - mu_male
        
        cov_male = cov_male + np.matmul(temp_matrix.transpose(), temp_matrix)
    
    cov_male = (1 / N_m) * cov_male
    
    # covariance matrix for QDA (female)
    cov_female = np.zeros((2,2))
    
    for j in range(0, len(female_data)):
        temp_matrix = np.zeros((1,2))
        temp_matrix[0] = female_data[j] - mu_female
        
        cov_female = cov_female + np.matmul(temp_matrix.transpose(), temp_matrix)
    
    cov_female = (1 / N_f) * cov_female

    # ----------------------------------------------------------------------- #
    # plotting LDA
    # ----------------------------------------------------------------------- #    
    
    # Criteria 1: Data points from training set
    
    lda_one = plt.figure(1)
    axes = lda_one.add_axes([0.1, 0.1, 0.8, 0.8])
    
    for each_blue_dot in male_data:
        lda_one_blue = plt.scatter(each_blue_dot[0], each_blue_dot[1], color = 'blue')
        
    for each_red_dot in female_data:
        lda_one_red = plt.scatter(each_red_dot[0], each_red_dot[1], color = 'red')
    
    # Criteria 2: Contours of conditional Gaussian distribution
    
    x_axis = np.arange(50, 80, 0.15)
    y_axis = np.arange(80, 280, 1)
    
    X_label, Y_label = np.meshgrid(x_axis, y_axis)    
    x_set = np.dstack((X_label, Y_label))
    x_set = np.reshape(x_set, (len(X_label)*len(Y_label) , 2))

    gauss_densities_male = util.density_Gaussian(mu_male, cov, x_set).reshape(X_label.shape[0], X_label.shape[1])
    gauss_densities_female = util.density_Gaussian(mu_female, cov, x_set).reshape(X_label.shape[0], X_label.shape[1])

    plt.contour(X_label, Y_label, gauss_densities_male)
    plt.contour(X_label, Y_label, gauss_densities_female)

    # Criteria 3: Boundary Line

    LHS_lda = ((-0.5) * np.matmul(mu_male.transpose(), np.matmul(np.linalg.inv(cov), mu_male)) + np.matmul(mu_male.transpose(), np.matmul(np.linalg.inv(cov), x_set.transpose()))).reshape(X_label.shape[0], X_label.shape[1])
    RHS_lda = ((-0.5) * np.matmul(mu_female.transpose(), np.matmul(np.linalg.inv(cov), mu_female)) + np.matmul(mu_female.transpose(), np.matmul(np.linalg.inv(cov), x_set.transpose()))).reshape(X_label.shape[0], X_label.shape[1])

    Z = LHS_lda - RHS_lda

    plt.contour(X_label, Y_label, Z, 0)

    axes.set_xlim([50, 80])
    axes.set_ylim([80, 280])
    
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Training Data (LDA)')
    plt.legend([lda_one_blue, lda_one_red],
                ['Male Data', 'Female Data'],
                scatterpoints = 1,
                loc = 'upper left')    
    plt.savefig("lda.pdf")
    # ----------------------------------------------------------------------- #
    # plotting QDA
    # ----------------------------------------------------------------------- #    
    
    # Criteria 1: Data points from training set
    
    qda_one = plt.figure(2)
    qda_axes = qda_one.add_axes([0.1, 0.1, 0.8, 0.8])
    
    for each_blue_dot in male_data:
        qda_one_blue = plt.scatter(each_blue_dot[0], each_blue_dot[1], color = 'blue')
        
    for each_red_dot in female_data:
        qda_one_red = plt.scatter(each_red_dot[0], each_red_dot[1], color = 'red')
    
    # Criteria 2: Contours of conditional Gaussian distribution
    
    x_axis = np.arange(50, 80, 0.15)
    y_axis = np.arange(80, 280, 1)
    
    X_label, Y_label = np.meshgrid(x_axis, y_axis)    
    x_set = np.dstack((X_label, Y_label))
    x_set = np.reshape(x_set, (len(X_label)*len(Y_label) , 2))

    gauss_densities_male = util.density_Gaussian(mu_male, cov_male, x_set).reshape(X_label.shape[0], X_label.shape[1])
    gauss_densities_female = util.density_Gaussian(mu_female, cov_female, x_set).reshape(X_label.shape[0], X_label.shape[1])

    plt.contour(X_label, Y_label, gauss_densities_male)
    plt.contour(X_label, Y_label, gauss_densities_female)
    
    # Criteria 3: Boundary Line

    x_points = np.arange(50, 80, 0.6)
    y_points = np.arange(80, 280, 4)
    
    X_points, Y_points = np.meshgrid(x_points, y_points)    
    x_bound = np.dstack((X_points, Y_points))
    x_bound = np.reshape(x_bound, (len(X_points)*len(Y_points) , 2))

    mu_m_vec = np.zeros((len(x_bound), 2))
    mu_f_vec = np.zeros((len(x_bound), 2))
    for i in range(0, len(x_bound)):
        mu_m_vec[i] = mu_male
        mu_f_vec[i] = mu_female
    
    m_det = 0.5 * np.log(np.linalg.det(cov_male))
    f_det = 0.5 * np.log(np.linalg.det(cov_female))
    
    x_mu_diff_m = x_bound - mu_m_vec
    x_mu_diff_f = x_bound - mu_f_vec
    
    x_mu_diff_m_T = x_mu_diff_m.transpose()
    x_mu_diff_f_T = x_mu_diff_f.transpose()
    
    cov_male_inverted = np.linalg.inv(cov_male)
    cov_female_inverted = np.linalg.inv(cov_female)

    result_m = np.matmul(cov_male_inverted, x_mu_diff_m_T)
    result_f = np.matmul(cov_female_inverted, x_mu_diff_f_T)

    result2_m = np.matmul(x_mu_diff_m, result_m) #.reshape(X_points.shape[0], X_points.shape[1])
    result2_f = np.matmul(x_mu_diff_f, result_f) #.reshape(X_points.shape[0], X_points.shape[1])

    # NOTE: DID NOT PLOT QUADRATIC BOUNDARY AS THE RESHAPING WAS THROWING ERRORS

#    Z = result2_m - result2_f
#    plt.contour(X_label, Y_label, Z, 0)
    
    qda_axes.set_xlim([50, 80])
    qda_axes.set_ylim([80, 280])
    
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.title('Height vs. Weight for Training Data (QDA)')
    plt.legend([qda_one_blue, qda_one_red],
                ['Male Data', 'Female Data'],
                scatterpoints = 1,
                loc = 'upper left')
    plt.savefig("qda.pdf")

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
   
    # Test the data and store it in a data structure
    lda_results = []
    qda_results = []
    
    # LDA Calculations
    LHS_lda = (-0.5) * np.matmul(mu_male.transpose(), np.matmul(np.linalg.inv(cov), mu_male)) + np.matmul(mu_male.transpose(), np.matmul(np.linalg.inv(cov), x.transpose()))
    RHS_lda = (-0.5) * np.matmul(mu_female.transpose(), np.matmul(np.linalg.inv(cov), mu_female)) + np.matmul(mu_female.transpose(), np.matmul(np.linalg.inv(cov), x.transpose()))
    
    # LDA Results
    for i in range(0, len(LHS_lda)):
        if LHS_lda[i] > RHS_lda[i]:
            lda_results.append(1)
        else:
            lda_results.append(2)
    
    # QDA Calculations
    mu_m_vec = np.zeros((len(x), 2))
    mu_f_vec = np.zeros((len(x), 2))
    for i in range(0, len(x)):
        mu_m_vec[i] = mu_male
        mu_f_vec[i] = mu_female
    
    m_det = 0.5 * np.log(np.linalg.det(cov_male))
    x_mu_diff_m = x - mu_m_vec
    x_mu_diff_m_T = x_mu_diff_m.transpose()
    cov_male_inverted = np.linalg.inv(cov_male)
    result_m = np.matmul(cov_male_inverted, x_mu_diff_m_T)
    result2_m = np.matmul(x_mu_diff_m, result_m) + m_det
    
    f_det = 0.5 * np.log(np.linalg.det(cov_female))
    x_mu_diff_f = x - mu_f_vec
    x_mu_diff_f_T = x_mu_diff_f.transpose()    
    cov_female_inverted = np.linalg.inv(cov_female)
    result_f = np.matmul(cov_female_inverted, x_mu_diff_f_T)
    result2_f = np.matmul(x_mu_diff_f, result_f) + f_det
        
    # QDA Results
    
    # Calculate error = (number of misses / total samples) * 100 %
    total_samples = len(x)
    lda_total_misses = 0
    qda_total_misses = 0
    
    for i in range(0, len(y)):
        if y[i] != lda_results[i]:
            lda_total_misses += 1
                
    mis_lda = lda_total_misses / total_samples
    mis_qda = qda_total_misses / total_samples
    
    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    

    
    
    

    
