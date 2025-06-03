#!/usr/bin/env python
# coding: utf-8

# In[196]:


    
def simulate(params = [0.5, 1, 0.01, 0.5], dimension1 = 'A', subject = 0, rng = None, seed = 999):
    # function to simulate data for the dimension learning task using the safrl model

    ''' input arguments
    params (list) - parameters of the model: learning rate, temperature, dimension primacy, dimension learning rate
    dimension1 (int) - first relevant stimulus dimension, A = 0, B = 1
    subject (int) - subject number
    rng (func) - set random number generator function to be used
    seed (int) - if no rng function provided, this seed will be used to generate new ones 
    '''

    ''' returns 
    dict:
    '''
    ## setup ## 

    # packages
    import numpy as np
    import pandas as pd
    import itertools

    # set random number generator
    rng = rng if rng else np.random.default_rng(seed)

    # get parameters
    alpha, beta, theta0, eps = params

    # set weights to their initial values, 0s for feature weights, and theta = dimension primacy parameter
    weights_0 = [np.array([0.0]*12).reshape(1,12), np.array([0.0]*12).reshape(1,12)]
    w_a, w_b, tmp_theta = weights_0[0].copy(), weights_0[1].copy(), np.array((theta0)).reshape(1,1)

    # task specifications, try to iterate over several designs later
    max_t = 50 # max trials
    pass_criteria = 8 # number of trials subejct must get correct in a row to pass to next block
    dimension1 = dimension1 # first relevant dimension
    stages = ['SD', 'CD', 'IDS_1', 'IDS_2', 'IDS_3', 'IDS_4', 'EDS', 'EDSR'] # block names

    # if dimension 1 is 0 then dimension A is firt
    if dimension1 == 'A':
        # stimuli vector, important to maintain order as specific indexes used later
        stimuli = ['A_01', 'A_02',
                   'A_01,B_01', 'A_02,B_01', 'A_01,B_02', 'A_02,B_02',
                   'A_03,B_03', 'A_04,B_03', 'A_03,B_04', 'A_04,B_04',
                   'A_05,B_05', 'A_06,B_05', 'A_05,B_06', 'A_06,B_06',
                   'A_07,B_07', 'A_08,B_07', 'A_07,B_08', 'A_08,B_08',
                   'A_09,B_09', 'A_10,B_09', 'A_09,B_10', 'A_10,B_10',
                   'A_11,B_11', 'A_12,B_11', 'A_11,B_12', 'A_12,B_12'
                   ]
        
        # feature with high reward probability for each block
        target_feature = ['A_01', 'A_01', 'A_03', 'A_05', 'A_07', 'A_09', 'B_11', 'B_12']

    else: # dimension B is first

        # stimuli vector, important to maintain order as specific indexes used later
        stimuli = ['B_01', 'B_02',
                   'A_01,B_01', 'A_02,B_01', 'A_01,B_02', 'A_02,B_02',
                   'A_03,B_03', 'A_04,B_03', 'A_03,B_04', 'A_04,B_04',
                   'A_05,B_05', 'A_06,B_05', 'A_05,B_06', 'A_06,B_06',
                   'A_07,B_07', 'A_08,B_07', 'A_07,B_08', 'A_08,B_08',
                   'A_09,B_09', 'A_10,B_09', 'A_09,B_10', 'A_10,B_10',
                   'A_11,B_11', 'A_12,B_11', 'A_11,B_12', 'A_12,B_12'
                   ]
        
        # feature with high reward probability for each block
        target_feature = ['B_01', 'B_01', 'B_03', 'B_05', 'B_07', 'B_09', 'A_11', 'A_12']


    # list of all features
    features = ['A_01', 'A_02', 'A_03', 'A_04', 'A_05', 'A_06', 'A_07', 'A_08', 'A_09', 'A_10', 'A_11', 'A_12',
                'B_01', 'B_02', 'B_03', 'B_04', 'B_05', 'B_06', 'B_07', 'B_08', 'B_09', 'B_10', 'B_11', 'B_12']

    # one hot encoded vector for each feature
    A, B = dict(zip(features[:12], np.identity(12))), dict(zip(features[12:], np.identity(12)))

    # task variables to be filled
    inputs = []
    feature_weights = [] # feature weights
    val = [] # stimulus values
    choice = [] # choice
    reward = [] # reward
    weights_a = [w_a.copy()] # weights for dimension A
    weights_b = [w_b.copy()] # weights for dimension B
    theta = [tmp_theta.copy()] # dimension bias
    pe = [] #  prediction error
    stim_hist = [] # stimuli history
    feedback = [] # trialwise feedback
    n_trials = [] # n trials per stage
    n_errors = [] # n errors per stage
    stages_attempted = []
    stages_passed = []

    # define sigmoid function
    def sigmoid(x):
        y = 1/(1+np.exp(-x))
        return(y)

    ## now start simulation ##

    # initalise trial number
    t = 0

    # iterate over trials and stages
    for stage in range(len(stages)):
        for stage_t in range(max_t):

            # first get stimuli to be presented 
            if (stage == 0):
                # generate random permutation of first two stim in stim list for stage 1
                s = list(rng.permutation(stimuli[0:2]))
                
            else:
                # select compound stimuli to be presented for all other stages
                if stage == 1:
                    n = 7
                    stim1 = 2
                elif stage == 2:
                    n = 15
                    stim1 = 6
                elif stage == 3:
                    n = 23
                    stim1 = 10
                elif stage == 4:
                    n = 31
                    stim1 = 14
                elif stage == 5:
                    n = 39
                    stim1 = 18
                elif stage in [6,7]:
                    n = 47
                    stim1 = 22

                # get flat list of stimuli from previous 3 and 6 trials
                prev_3 = [item for sublist in stim_hist[-3:] for item in sublist]
                prev_6 = [item for sublist in stim_hist[-6:] for item in sublist]

                # check whether stimuli break rules
                for x in stimuli[stim1:stim1+2]: 
                    # stimuli combination should not repeat more than 3 times in a row or more than 4 in 6 consecutive trials
                    if prev_3.count(x) >= 3 or prev_6.count(x) >= 4: # if so, select other stimuli combination for this trial
                        rep = 1
                        stimremain = [stim for stim in stimuli[stim1:stim1 +4] if stim not in [x, stimuli[n - stimuli.index(x)]]]
                        s[0] = rng.choice(stimremain)
                        s[1] = stimuli[n - stimuli.index(s[0])]
                        break
                    else:
                        rep = 0

                if rep == 0: # if not, select stimuli combination at random for this trial
                    s[0] = rng.choice(stimuli[stim1:stim1 +4])
                    s[1] = stimuli[n - stimuli.index(s[0])]
            stim_hist.append(s.copy())

            # get features of stimuli presented
            if (stage == 0):
                A1,A2 = (A[str(stim_hist[t][0])].reshape(12,1), A[str(stim_hist[t][1])].reshape(12,1)) if dimension1 == 'A' else (np.zeros((12,1)),np.zeros((12,1)))
                B1,B2 = (np.zeros((12,1)),np.zeros((12,1))) if dimension1 == 'A' else (B[str(stim_hist[t][0])].reshape(12,1), B[str(stim_hist[t][1])].reshape(12,1))
            elif stage in [1,2,3,4,5,6,7]:
                st1,st2 = stim_hist[t][0].replace(',',' '),stim_hist[t][1].replace(',',' ')
                a1, b1 = st1.split()
                a2, b2 = st2.split()
                A1,A2 = (A[a1].reshape(12,1), A[a2].reshape(12,1))
                B1,B2 = (B[b1].reshape(12,1), B[b2].reshape(12,1))

            # list of lists on each trial (sublist for each stimulus)
            inputs.append([[A1,B1],[A2,B2]])

            # FEEDFORWARD to estimate stimulus values
            if stage == 0: #simple feature reinforcement learning for stages 1 
                v1,v2 = (np.dot(w_a, A1),np.dot(w_a, A2)) if dimension1 == 'A' else                            (np.dot(w_b, B1),np.dot(w_b, B2)) # input multiplied by weight = stimulus value
                val.append(np.concatenate((v1,v2)))
                feature_weights.append([[0,0],[0,0]])
            else: # dimension weights added for other stages
                # inputs multiplied by their weights, weighted by dimension weight = stimulus values
                feat_w1,feat_w2,feat_w3,feat_w4 = np.dot(w_a, A1),np.dot(w_b, B1),np.dot(w_a, A2),np.dot(w_b, B2)
                feature_weights.append([[feat_w1, feat_w2],[feat_w3, feat_w4]])
                v1,v2 = sigmoid(tmp_theta)*feat_w1+(1-sigmoid(tmp_theta))*feat_w2, sigmoid(tmp_theta)*feat_w3+(1-sigmoid(tmp_theta))*feat_w4
                val.append(np.concatenate((v1,v2)))

            # make choice using softmax with predicted values    
            ev = np.exp(beta*val[t])
            sev = sum(ev)
            p = ev/sev
            choice.append(stim_hist[t][0]) if rng.uniform() < p[0] else choice.append(stim_hist[t][1])

            # get feedback on choice 
            R1 = 1 if target_feature[stage] in stim_hist[t][0] else -1
            R2 = 0 - R1 # reward for other stimulus is inferred to be the opposite
            reward.append([R1,R2])

            feedback.append(reward[t][stim_hist[t].index(choice[t])])
            pe.append(reward[t][stim_hist[t].index(choice[t])] - val[t][stim_hist[t].index(choice[t])]) # calculate prediction error

            # Update Weights ================
            if stage in [0]: #simple RL on feature weights
                w_a += alpha * (reward[t][0] - val[t][0]) * inputs[t][0][0].T
                w_b += alpha * (reward[t][0] - val[t][0]) * inputs[t][0][1].T
                w_a += alpha * (reward[t][1] - val[t][1]) * inputs[t][1][0].T
                w_b += alpha * (reward[t][1] - val[t][1]) * inputs[t][1][1].T

                weights_a.append(w_a.copy())
                weights_b.append(w_b.copy())
                theta.append(tmp_theta.copy())

            else: # backpropagation for dimension and feature weights

                ### Phase1 

                ### for dimension weight

                ## we need to get the gradient of the loss with repsect to theta
                # done so by differentiating the squared error loss with respect to theta, d_L/d_theta
                # this is a composite function so requres chain rule
                # d_L/d_theta = d_L/d_y_pred x d_y_pred/d_theta

                ## first get the gradient of the loss function with respect to y_pred d_L/d_y_pred
                # this is a composite function so to differentiate we must apply chain rule again
                # let's call the difference between outcome and prediction (loss) u = y - y_pred
                # the loss function used here is the squared error loss L = 1/2(u)^2
                # chain rule states d_L/d_y_pred = d_L/d_u * d_u/d_y_pred
                # 1. outer function, d_L/d_u = u
                # 2. inner function, d_u/d_y_pred = d/d_y_pred(y-y_pred) = d/d_y_pred(y) - d/d_y_pred(y_pred) = 0 - 1 = -1
                # because y is a constant and y_pred is the variable we are differentitating with respect to
                # 3. multiply output of 1 and 2, u*-1 = y_pred - y or ev[chosen] - R

                '''
                The gradient tells us how the loss changes with y_pred
                if y-y_pred is positive, the gradient is negative, 
                which means increasing y_pred will decrease the loss. 
                '''
                dcost_dV = val[t][stim_hist[t].index(choice[t])] - reward[t][stim_hist[t].index(choice[t])]

                # now lets get d_y_pred/d_theta
                # for the chosen stim we know y_pred = sigmoid(theta)*f(a) + 1-simoid(theta)*f(b)
                # now we have to differnetiate y_pred with respect to theta
                # the derivation of sigmoid(x) with respect to x is sigmoid(x) * 1-sigmoid(x)
                # so the d_y_pred/d_theta = (sigmoid(theta) * (1-sigmoid(theta))) * (f(a)-f(b))
                # hence sigmoid(theta)*(1-sigmoid_theta)*f(a) - sigmoid(theta)*(1-sigmoid_theta)*f(b)
                dV_dtheta = sigmoid(tmp_theta)*(1-sigmoid(tmp_theta))*feature_weights[t][stim_hist[t].index(choice[t])][0] -                                 sigmoid(tmp_theta)*(1-sigmoid(tmp_theta))*feature_weights[t][stim_hist[t].index(choice[t])][1]

                # now we multiply d_cost/d_y_pred * d_y_pred/d_theta = d_cost/d_theta
                dcost_theta = np.dot(dcost_dV, dV_dtheta.T) 

                ### for feature weights

                # this is basically same process as above
                # here we differentiate cost function L = 1/2(y-y_pred)^2 with resepect to to y_pred (stim value)
                dcost_dV1 = val[t][0] - reward[t][0] 
                dcost_dV2 = val[t][1] - reward[t][1]

                # now we have to differentiate y_pred (V) with repsect to the feature weights
                # given that V = sigmoid(theta) * f(A) + (1-sigmoid(theta)) * f(B)
                # if we differentiate V with respect to f(A)
                # the derivative of sigmoid(theta) * f(A) = sigmoid(theta), while the derivative of 1-sigmoid(theta) * f(b) = 0, so together is just sigmoid(theta)
                # conversely dv_df(B) =  1-sigmoid(theta)
                dV_dah1 = sigmoid(tmp_theta)
                dV_dah2 = 1-sigmoid(tmp_theta)

                # now we just multiply together to satisfy chain rule
                dcost_dah1 = np.dot(dV_dah1.T, dcost_dV1) #V(S1, F(A))
                dcost_dah2 = np.dot(dV_dah2.T, dcost_dV1) #V(S1, F(B))
                dcost_dah3 = np.dot(dV_dah1.T, dcost_dV2) #V(S2, F(A))
                dcost_dah4 = np.dot(dV_dah2.T, dcost_dV2) #V(S2, F(B))

                # Phase 2
                # now we multiply with one-hot encoded input vector 
                dah1_dwh1 = inputs[t][0][0]
                dah2_dwh2 = inputs[t][0][1]
                dah3_dwh1 = inputs[t][1][0]
                dah4_dwh2 = inputs[t][1][1]
                dcost1_wh1 = np.dot(dcost_dah1, dah1_dwh1.T)
                dcost1_wh2 = np.dot(dcost_dah2, dah2_dwh2.T)
                dcost2_wh1 = np.dot(dcost_dah3, dah3_dwh1.T)
                dcost2_wh2 = np.dot(dcost_dah4, dah4_dwh2.T)

                # Update Weights ================

                # updates are negative as the derivative with respect to the weights are negative when we want to increase 
                w_a -= alpha * dcost1_wh1
                w_b -= alpha * dcost1_wh2
                w_a -= alpha * dcost2_wh1
                w_b -= alpha * dcost2_wh2
                tmp_theta -= eps * dcost_theta


                weights_a.append(w_a.copy())
                weights_b.append(w_b.copy())
                theta.append(tmp_theta.copy())

            # terminate stage and move on if select correct stimulus 8 times in a row on new stage
            if (stage_t >= (pass_criteria-1))  & (all(target_feature[stage] in x for x in choice[-pass_criteria:])):
                n_trials.append(stage_t + 1) # n trials for current stage
                n_errors.append(feedback[-(stage_t+1):].count(-1))
                stages_attempted.append(stages[stage]) # add stage name
                stages_passed.append(True) # add that participant passed this stage
                t+=1
                break
            elif stage_t == max_t - 1: # or terminate stage and task if reached maxed trials
                n_trials.append(stage_t + 1)
                n_errors.append(feedback[-(stage_t+1):].count(-1))
                stages_attempted.append(stages[stage])
                stages_passed.append(False)
                t+=1

                results = {'pe':pe, 'choice': choice, 'val': val, 'feature_weights': feature_weights, 'reward': reward, 'feedback': feedback, 'stim_hist': stim_hist, 'inputs': inputs, 'weights_a':weights_a,\
                           'weights_b': weights_b, 'theta': theta, 'alpha': alpha, 'beta': beta, 'theta_0': theta0.copy(), 'eps': eps, 'weights_0': weights_0,\
                            'dimension1': dimension1, 'subject': subject, 'stage':list(itertools.chain(*[[stages[x]]*n_trials[x] for x in range(len(n_trials))])),\
                            'n_trials': n_trials, 'n_erros' :n_errors, 'stages_attempted' :stages_attempted, 'stages_passed': stages_passed}

                return results
            
            t+=1
    results = {'pe':pe, 'choice': choice, 'val': val, 'feature_weights': feature_weights, 'reward': reward, 'feedback': feedback, 'stim_hist': stim_hist, 'inputs': inputs, 'weights_a':weights_a,\
                'weights_b': weights_b, 'theta': theta, 'alpha': alpha, 'beta': beta, 'theta_0': theta0.copy(), 'eps': eps, 'weights_0': weights_0,\
                'dimension1': dimension1, 'subject': subject, 'stage':list(itertools.chain(*[[stages[x]]*n_trials[x] for x in range(len(n_trials))])),\
                'n_trials': n_trials, 'n_erros' :n_errors, 'stages_attempted' :stages_attempted, 'stages_passed': stages_passed}

    return results

def llIED(params,args):
    """Calculate (log) likelihood for subject data given parameter values. 

    Parameters:
    params (list): list of parameters in order [alpha: learning rate, beta: choice determinism, theta0: dimension primacy]. 
    args (list): list of arguments for function, [R: list of rewards, choice: list of stimulus choices, S: list of stimuli seen , dimension1: string of first relevant dimension, likelihood: boolean - False if calculating log likelihood, u: prior mean, v2: prior covariance].

    Returns:
    float: if calculating log likehood log likelihood for ML or log posterior for EM
    or
    tuple: if calculating likelihood for iBIC or alpt, 
    (likelihood - total data likelihood for this subject,
    avg_likelihood - average likelihood per trial for this subject)
   
   """
    import numpy as np
    import warnings
    from scipy.special import logsumexp

    #warnings.simplefilter('ignore', RuntimeWarning)
    
    nP = len(params)

    alpha, beta, theta0, eps = params
    aeby = np.asarray(params.copy()).reshape(nP,1)

    R,choice,S,dimension1,likelihood = args[:5]
    
    if not likelihood:

        # transform parameters
        alpha = 1/(1 + np.exp(-aeby[0]))
        beta = np.exp(aeby[1])
        theta0 = np.array((aeby[2])).reshape(1,1)
        eps = 1/(1 + np.exp(-aeby[3]))

    if len(args) > 5:
        # we are doing EM
        u = args[5]
        v2 = args[6]
        u = u.reshape(nP,1)

        PP = aeby - u
        L = 0.5*(PP.T @ np.linalg.pinv(v2) @ PP)
        LP = -np.log(2*np.pi) - 0.5*np.log(np.linalg.det(v2)) - L
        NLP = -LP[0] # calculate negative log liklihood of drawing these parameters from a given prior

    fs = ['A_01', 'A_02', 'A_03', 'A_04', 'A_05', 'A_06', 'A_07', 'A_08', 'A_09', 'A_10', 'A_11', 'A_12',
                'B_01', 'B_02', 'B_03', 'B_04', 'B_05', 'B_06', 'B_07', 'B_08', 'B_09', 'B_10', 'B_11', 'B_12']
    lines,shapes = dict(zip(fs[:12], np.identity(12))),dict(zip(fs[12:], np.identity(12)))

    # initalise empty lists
    AH,V,inputs = [],[],[]
    l,ll = [0]*2,0
    like = []
    theta_t = []
    
    wh0 = [np.array([0.0]*12).reshape(1,12),np.array([0.0]*12).reshape(1,12)]
    wh1 = wh0[0].copy()
    wh2 = wh0[1].copy()
    theta = np.array((aeby[2])).reshape(1,1)

    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return(s)

    for t in range(len(choice)):
        if len(S[t][0]) == 4:
            if dimension1 == 'A':
                l1 = lines[str(S[t][0])].reshape(12,1)
                l2 = lines[str(S[t][1])].reshape(12,1)
                s1,s2 = np.zeros((12,1)),np.zeros((12,1))
            else: 
                s1 = shapes[str(S[t][0])].reshape(12,1)
                s2 = shapes[str(S[t][1])].reshape(12,1)
                l1,l2 = np.zeros((12,1)),np.zeros((12,1))
            inputs.append([[l1,s1],[l2,s2]])

            # FEEDFORWARD to estimate stimulus values
            V1,V2 = (np.dot(wh1, l1),np.dot(wh1, l2)) if dimension1 == 'A' else                            (np.dot(wh2, s1),np.dot(wh2, s2))
            V.append(np.concatenate((V1,V2)))
            AH.append([[0,0],[0,0]])
            vmax = beta*np.amax(V[t])

            #l = beta*(V[t][S[t].index(choice[t])] - vmax) - np.log(sum((np.exp(beta*(V[t][x]-vmax)) for x in range(len(S[t])))))
            l = beta * (V[t][S[t].index(choice[t])] - vmax) - logsumexp([beta * (V[t][x] - vmax) for x in range(len(S[t]))])
            ll += l.copy()

            if likelihood == True:
                ev = np.exp(beta*V[t])
                sev = sum(ev)
                p = ev/sev

                like.append(p[S[t].index(choice[t])])

            # Update Weights ================

            wh1 += alpha * (R[t][0] - V[t][0]) * inputs[t][0][0].T
            wh2 += alpha * (R[t][0] - V[t][0]) * inputs[t][0][1].T
            wh1 += alpha * (R[t][1] - V[t][1]) * inputs[t][1][0].T
            wh2 += alpha * (R[t][1] - V[t][1]) * inputs[t][1][1].T

            theta_t.append(theta.copy())

        else:
            st1,st2 = S[t][0].replace(',',' '),S[t][1].replace(',',' ')
            if dimension1 == 'A':
                line1, shape1 = st1.split()
                line2, shape2 = st2.split()
            else:
                line1, shape1 = st1.split()
                line2, shape2 = st2.split()
            l1,s1 = lines[line1].reshape(12,1),shapes[shape1].reshape(12,1)
            l2,s2 = lines[line2].reshape(12,1),shapes[shape2].reshape(12,1)
            inputs.append([[l1,s1],[l2,s2]])

            # FEEDFORWARD to estimate stimulus values
            ah1,ah2,ah3,ah4 = np.dot(wh1, l1),np.dot(wh2, s1),np.dot(wh1, l2),np.dot(wh2, s2)
            AH.append([[ah1,ah2],[ah3,ah4]])

            V1,V2 = sigmoid(theta)*ah1+(1-sigmoid(theta))*ah2, sigmoid(theta)*ah3+(1-sigmoid(theta))*ah4
            V.append(np.concatenate((V1,V2)))

            vmax = beta*np.amax(V[t])
            
            l = beta * (V[t][S[t].index(choice[t])] - vmax) - logsumexp([beta * (V[t][x] - vmax) for x in range(len(S[t]))])
            #l = beta*(V[t][S[t].index(choice[t])] - vmax) - np.log(sum((np.exp(beta*(V[t][x]-vmax)) for x in range(len(S[t])))))
            ll += l.copy()

            if likelihood == True:
                ev = np.exp(beta*V[t])
                sev = sum(ev)
                p = ev/sev

                like.append(p[S[t].index(choice[t])])

            dcost_dV = V[t][S[t].index(choice[t])] - R[t][S[t].index(choice[t])]

            dV_dtheta = sigmoid(theta)*(1-sigmoid(theta))*AH[t][S[t].index(choice[t])][0] -                                 sigmoid(theta)*(1-sigmoid(theta))*AH[t][S[t].index(choice[t])][1]

            dcost_theta = np.dot(dcost_dV, dV_dtheta.T)

            dcost_dV1 = V[t][0] - R[t][0]
            dcost_dV2 = V[t][1] - R[t][1]

            dV_dah1 = sigmoid(theta)
            dV_dah2 = 1-sigmoid(theta)

            dcost_dah1 = np.dot(dV_dah1.T, dcost_dV1)
            dcost_dah2 = np.dot(dV_dah2.T, dcost_dV1)
            dcost_dah3 = np.dot(dV_dah1.T, dcost_dV2)
            dcost_dah4 = np.dot(dV_dah2.T, dcost_dV2)

            # Phase 2
            dah1_dwh1 = inputs[t][0][0]
            dah2_dwh2 = inputs[t][0][1]
            dah3_dwh1 = inputs[t][1][0]
            dah4_dwh2 = inputs[t][1][1]
            dcost1_wh1 = np.dot(dcost_dah1, dah1_dwh1.T)
            dcost1_wh2 = np.dot(dcost_dah2, dah2_dwh2.T)
            dcost2_wh1 = np.dot(dcost_dah3, dah3_dwh1.T)
            dcost2_wh2 = np.dot(dcost_dah4, dah4_dwh2.T)

            # Update Weights ===============

            wh1 -= alpha * dcost1_wh1
            wh2 -= alpha * dcost1_wh2
            wh1 -= alpha * dcost2_wh1
            wh2 -= alpha * dcost2_wh2
            theta -= eps * dcost_theta

            theta_t.append(theta.copy())

    if likelihood == True:
        return (np.prod(like), np.mean(like), like, -ll, theta_t)

    if len(args) > 6:
        llEM = -ll + NLP 
        return llEM
    else:
        return -ll

def fit(params):
    """Optimise (log) likelihood for subject data given parameter values. 

    Parameters:
    params (list): list of parameters to pass to minimisation function, depends on whether using EM or ML. 

    Returns:
    tuple: of (minimisation result, number of minimisation attempts) if using EM
    or
    array: of best-fitting parameters if using ML
   
   """
    #read in appropriate data and set up
    import scipy.optimize
    import numpy as np
    import warnings
    
    #warnings.simplefilter('ignore', RuntimeWarning)

    if len(params) > 6:
        # we are using EM
        args = params[:-3]
        rng = params[-1]
        var = 1
        while var >= 1:
            for x in range(50):
                if x == 0:
                    m = params[-3]
                else:
                    m = params[-3]+ x*0.1*np.matmul(rng.standard_normal((1,(len(m)))), scipy.linalg.sqrtm(np.linalg.inv(0.1*np.identity((len(m))))))
                xopt = scipy.optimize.minimize(fun=llIED, x0=np.ravel(list(m)),args = (args))
                if xopt.success:
                    warnings.resetwarnings()
                    return([xopt,var])
                else:
                    continue             
                var+=1
            warnings.resetwarnings()

    else:
        # we are using ML
        args = params[:-1]
        m = params[-1]   
        # minimise log likelihood for subject data and given parameters
        xopt = scipy.optimize.minimize(fun=llIED, x0=list(m),args =  (args))
        params =  list(xopt['x'])
        warnings.resetwarnings()
        return (params)


# In[ ]:




