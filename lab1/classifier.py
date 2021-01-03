import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import decimal

def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here

    # get word frequency for each file in each category
    spam_word_dict = util.get_word_freq(file_lists_by_category[0])
    ham_word_dict = util.get_word_freq(file_lists_by_category[1])

    # creating set to store all words in training files (i.e. master dictionary, W)
    w_list = set()

    # inputting words from spam and ham lists into the W list
    for key in spam_word_dict:
        w_list.add(key)

    for key in ham_word_dict:
        w_list.add(key)

    # numerical values required for p_d and q_d estimates
    D = len(w_list)
    total_spam_words = 0
    total_ham_words = 0
    
    for i in spam_word_dict:
        total_spam_words += spam_word_dict[i]
    for j in ham_word_dict:
        total_ham_words += ham_word_dict[j]
    
    # creating the dictionaries that are to be returned
    p_d_dict = dict()
    q_d_dict = dict()

    for word in w_list:
        p_d = 0
        q_d = 0
        
        if word in spam_word_dict:
            p_d = (spam_word_dict[word] + 1) / (total_spam_words + D)
        else:
            p_d = 1 / (total_spam_words + D)

        if word in ham_word_dict:
            q_d = (ham_word_dict[word] + 1) / (total_ham_words + D)
        else:
            q_d = 1 / (total_ham_words + D)


        p_d_dict[word] = p_d
        q_d_dict[word] = q_d
    
    probabilities_by_category = (p_d_dict, q_d_dict)
    return probabilities_by_category

def classify_new_email(filename,probabilities_by_category,prior_by_category, buffer):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    probabilities_by_category: output of function learn_distributions
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here

    # accumulating the information required to execute the MAP rule    
    
    likelihood_spam = 0
    likelihood_ham = 0
    
    x_vector = util.get_word_freq([filename])
    
    for each_word in x_vector:
        if each_word in probabilities_by_category[0]:
            likelihood_spam += np.log(probabilities_by_category[0][each_word]) * x_vector[each_word]
        if each_word in probabilities_by_category[1]:
            likelihood_ham += np.log(probabilities_by_category[1][each_word]) * x_vector[each_word]
    
    LHS = likelihood_spam - likelihood_ham
    RHS = np.log(prior_by_category[1]) - np.log(prior_by_category[0]) + buffer

    # MAP rule
    email_type = ""
    
    if LHS > RHS:
        email_type = 'spam'
    else:
        email_type = 'ham'

    # calculating the posterior probabilities for X
    posterior_probabilities = []    
    
    p_x = likelihood_spam + np.log(prior_by_category[0]) + likelihood_ham + np.log(prior_by_category[1])
    
    posterior_probabilities.append(likelihood_spam + np.log(prior_by_category[0]) - np.log(p_x))
    posterior_probabilities.append(likelihood_ham + np.log(prior_by_category[1]) - np.log(p_x))
        
    classify_result = (email_type, posterior_probabilities)
    
    return classify_result

if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    # Classify emails from testing set and measure the performance
    for filename in (util.get_files_in_folder(test_folder)):
        # Classify
        label,log_posterior = classify_new_email(filename,
                                                 probabilities_by_category,
                                                 priors_by_category,
                                                 0)
        
        # Measure performance (the filename indicates the true label)
        base = os.path.basename(filename)
        true_index = ('ham' in base) 
        guessed_index = (label == 'ham')
        performance_measures[int(true_index), int(guessed_index)] += 1

    template="You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
    # Correct counts are on the diagonal
    correct = np.diag(performance_measures)
    # totals are obtained by summing across guessed labels
    totals = np.sum(performance_measures, 1)
    print(template % (correct[0],totals[0],correct[1],totals[1]))
    
    
    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve
    
    buffer = [-5, 0, 5, 10, 15, 20, 25, 35, 40, 49]
    errors = []
    
    for i in buffer:
        new_performance_measures = np.zeros([2,2])
        
        # Classify emails from testing set and measure the performance
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label,log_posterior = classify_new_email(filename,
                                                     probabilities_by_category,
                                                     priors_by_category,
                                                     i)
            
            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base) 
            guessed_index = (label == 'ham')
            new_performance_measures[int(true_index), int(guessed_index)] += 1
    
        # Correct counts are on the diagonal
        new_correct = np.diag(new_performance_measures)
        # totals are obtained by summing across guessed labels
        new_totals = np.sum(new_performance_measures, 1)

        errors.append((new_totals[0] - new_correct[0], new_totals[1] - new_correct[1]))
        plt.scatter(new_totals[0] - new_correct[0], new_totals[1] - new_correct[1], color = 'blue')

    # trade-off curve
    plt.xlabel('Type 1 Errors')
    plt.ylabel('Type 2 Errors')
    plt.title('Type 2 vs Type 1 Errors using New Decision Rule')
        
    plt.savefig("nbc.pdf")
