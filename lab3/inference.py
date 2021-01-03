import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 
    
    for k in range(0, num_time_steps):
        forward_messages[k] = rover.Distribution()
        backward_messages[k] = rover.Distribution()
        marginals[k] = rover.Distribution()
    
    # TODO: Compute the forward messages
    
    # initialization:
    pi_0 = prior_distribution
    for i in pi_0:
        z0_obs = observation_model(i)
        for j in z0_obs:
            if (observations[0] == None):
                forward_messages[0][i] = pi_0[i] * 1
            
            elif (j == observations[0]):
                forward_messages[0][i] = pi_0[i] * z0_obs[j]

    forward_messages[0].renormalize()    

    # recursion
    for n in range(1, num_time_steps):
        A = dict()
        
        for i in forward_messages[n-1]: # [0]
            temp = transition_model(i)
            for j in temp:
                if j in A.keys():
                    A[j] += temp[j] * forward_messages[n-1][i]
                else:
                    A[j] = temp[j] * forward_messages[n-1][i]
        
        for i in A: # all the z_i's
            zi_obs = observation_model(i)
            for j in zi_obs:
                if (observations[n] == None):
                    forward_messages[n][i] = A[i] * 1
                    
                elif (j == observations[n]):
                    forward_messages[n][i] = A[i] * zi_obs[j]

        forward_messages[n].renormalize()
    
    # TODO: Compute the backward messages
   
    # initialization
    for h in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][h] = 1 # Beta(z_n)

    backward_messages[num_time_steps - 1].renormalize()

    # recursion
    for n in range(num_time_steps - 2, -1, -1):
        A = dict()
        
        for i in backward_messages[n + 1]:
            zi_obs = observation_model(i) # p((x_n, y_n)|z_n)
            for j in zi_obs:
                if (observations[n + 1] == None): 
                     A[i] = backward_messages[n + 1][i] * 1
                
                elif (j == observations[n + 1]):
                     A[i] = backward_messages[n + 1][i] * zi_obs[j]
        
        for i in A:
            for all_h in all_possible_hidden_states:
                temp_transition = transition_model(all_h)
                for k in temp_transition:
                    if (k == i and all_h in backward_messages[n].keys()):
                        backward_messages[n][all_h] += temp_transition[k] * A[i]
                    elif (k == i and all_h not in backward_messages[n].keys()):
                        backward_messages[n][all_h] = temp_transition[k] * A[i]
        backward_messages[n].renormalize()
    
    # TODO: Compute the marginals 
    for n in range(0, num_time_steps):
        for fm in forward_messages[n]:
            if fm in backward_messages[n].keys():
                marginals[n][fm] = forward_messages[n][fm] * backward_messages[n][fm]
        marginals[n].renormalize()
    
    # Determine the most probable state by maximizing each marginal to determine
    # a sequence of the most likely states
    FB_estimated_states = [None] * num_time_steps
    
    for i in range(0, num_time_steps):
        p_max = -1
        for k in marginals[i]:
            if marginals[i][k] > p_max:
                p_max = marginals[i][k]
                FB_estimated_states[i] = k
    
    return marginals, FB_estimated_states

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    estimated_hidden_states = [None] * num_time_steps
    
    messages = [None] * num_time_steps
    for m in range(0, num_time_steps):
        messages[m] = rover.Distribution()

    # initialization
    pi_0 = prior_distribution
    for i in pi_0:
        z0_obs = observation_model(i)
        for j in z0_obs:
            if (observations[0] == None):
                messages[0][i] = np.log(pi_0[i])
            elif (j == observations[0]):
                messages[0][i] = np.log(pi_0[i]) + np.log(z0_obs[j])
    
    
    # recursion
    z = [None] * num_time_steps
    
    for n in range(0, num_time_steps - 1):
        z[n] = dict()
        
        for z_i in messages[n]:
            trans_states = transition_model(z_i) # p(z_i+1 | z_i)
            for j in trans_states:
                if j not in z[n].keys():
                    z[n][j] = [(z_i, np.log(trans_states[j]) + messages[n][z_i])]       
                else:
                    z[n][j].append((z_i, np.log(trans_states[j]) + messages[n][z_i]))

        # Determine the z_i that transitions into z_i+1 with the highest probability
        for z_i_one in z[n]:
            p_max = -np.Infinity
            z_i_max = ()
            for each in z[n][z_i_one]:                
                if each[1] >= p_max:
                    p_max = each[1]
                    z_i_max = each[0]
            z[n][z_i_one] = [(z_i_max, p_max)]
    
        # For each z_i+1, determine p((x_i+1, y_i+1) | z_i+1)
        for z_i_one in z[n]:
            obs = observation_model(z_i_one)
            for k in obs:
                if (observations[n + 1] == None):
                    messages[n + 1][z_i_one] = np.log(1) + z[n][z_i_one][0][1]
                elif (k == observations[n + 1]):
                    messages[n + 1][z_i_one] = np.log(obs[k]) + z[n][z_i_one][0][1]
    
    # Backtracking to determine z_i from z_i+1
    # Determine the z_N-1 that maximizes the probability at the end of the chain
    new_p_max = -np.Infinity
    for i in messages[num_time_steps - 1]:
        if messages[num_time_steps - 1][i] >= new_p_max:
            new_p_max = messages[num_time_steps - 1][i]
            estimated_hidden_states[num_time_steps - 1] = i    

    # Use this z_i+1 to backtrack and determine the z_i that led to it
    for n in range(num_time_steps - 1, 0, -1):
        for j in z[n - 1]:
            if (j == estimated_hidden_states[n]):
                estimated_hidden_states[n - 1] = z[n - 1][j][0][0]

    return estimated_hidden_states

def ErrorProb(true_hidden_states,
              FB_estimated_states,
              Viterbi_estimated_states):
    
    error_probabilities = []
    
    # Determine P_e_tilde (error of the Viterbi state prediction)
    sum_tilde = 0
    sum_hat = 0
    
    for i in range(0, 100):
        if (Viterbi_estimated_states[i] == true_hidden_states[i]):
            sum_tilde += 1           
            
        if (FB_estimated_states[i] == true_hidden_states[i]):
            sum_hat += 1

    error_probabilities.append(1 - (sum_tilde / 100))
    error_probabilities.append(1 - (sum_hat / 100))
    
    return error_probabilities

def invalidSequence(estimated_hidden_states,
                    transition_model):
    sequence = []
    
    i = 0
    valid = True
    
    while (i < num_time_steps - 1):
        transition = transition_model(estimated_hidden_states[i])
        count_possible_states = len(transition)
        
        for k in transition:
            if (estimated_hidden_states[i + 1] == k):
                break
            else:
                valid = False
            count_possible_states -= 1
        
        # k has reached the last element in the transition probabilities
        if (count_possible_states == 0 and valid == False):
            valid = False
            sequence.append((estimated_hidden_states[i], i))
            sequence.append((estimated_hidden_states[i + 1], i + 1))
            break
        
        i = i + 1
        
    return sequence

if __name__ == '__main__':
   
    enable_graphics = False
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals, FB_estimated_states = forward_backward(all_possible_hidden_states,
                                     all_possible_observed_states,
                                     prior_distribution,
                                     rover.transition_model,
                                     rover.observation_model,
                                     observations)
    print('\n')
   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')
    
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    print('\n')
    
    print("Error probabilities for Question 4:\n")
    probabilities = ErrorProb(hidden_states,
                              FB_estimated_states,
                              estimated_states)
    
    print("P_e for Viterbi sequence = %.2f" % probabilities[0])
    print("P_e for forward-backward sequence = %.2f" % probabilities[1])
    
    
    print('\n')

    print("Invalid sequence of states generated by the forward-backward algorithm:")
    
    invalid_sequence = invalidSequence(FB_estimated_states,
                                       rover.transition_model)
    
    if missing_observations:
        print('\n')

        print("z_" + str(invalid_sequence[0][1]) + " = " + str(invalid_sequence[0][0]))
        print("z_" + str(invalid_sequence[1][1]) + " = " + str(invalid_sequence[1][0]))
    
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
