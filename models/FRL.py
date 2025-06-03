#!/usr/bin/env python
# coding: utf-8

# In[196]:


    
def simulate(params = [0.5, 1], dimension1 = 'A', subject = 0, rng = None, seed = 999):
    # function to simulate data for the dimension learning task using the safrl model

    ''' input arguments
    params (list) - parameters of the model: learning rate, temperature
    dimension1 (int) - first relevant stimulus dimension, A = 0, B = 1
    subject (int) - subject number
    rng (func) - set random number generator function to be used
    seed (int) - if no rng function provided, this seed will be used to generate new ones 
    '''

    ''' returns 
    dict:
    '''

    # packages
    import numpy as np
    import pandas as pd
    import itertools

    # set random number generator
    rng = rng if rng else np.random.default_rng(seed)

    # get parameters
    alpha, beta = params

    # set weights to their initial values, 0s for feature weights, and theta = dimension primacy parameter
    weights_0 = [np.array([0.0]*12).reshape(1,12), np.array([0.0]*12).reshape(1,12)]
    w_a, w_b = weights_0[0].copy(), weights_0[1].copy()

    # task specifications, try to iterate over several designs later
    max_t = 50 # max trials
    pass_criteria = 8 # number of trials subejct must get correct in a row to pass to next block
    dimension1 = dimension1 # first relevant dimension
    stages = ['SD', 'CD', 'IDS_1', 'IDS_2', 'IDS_3', 'IDS_4', 'EDS', 'EDSR'] # block names

    # if dimension 1 is 0 then dimension A is firt
    if dimension1 == 'A':
        # stimulus vector, important to maintain order as specific indexes used later
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

        # stimulus vector, important to maintain order as specific indexes used later
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

    # identity matrix for each dimension
    A, B = dict(zip(features[:12], np.identity(12))), dict(zip(features[12:], np.identity(12)))

    # task variables to be filled
    inputs = []
    feature_weights = [] # feature weights
    val = [] # stimulus values
    choice = [] # choice
    reward = [] # reward
    weights_a = [w_a.copy()] # weights for dimension A
    weights_b = [w_b.copy()] # weights for dimension B
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

    # now start simulation
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

            else: # dimension weights added for other stages
                # inputs multiplied by their weights, weighted by dimension weight = stimulus values
                v1, v2 = np.dot(w_a, A1) + np.dot(w_b, B1), np.dot(w_a, A2) + np.dot(w_b, B2)
            val.append(np.concatenate((v1,v2)))

            # make choice using softmax with predicted values    
            ev = np.exp(beta*val[t])
            sev = sum(ev)
            p = ev/sev
            choice.append(stim_hist[t][0]) if rng.uniform() < p[0] else choice.append(stim_hist[t][1])

            # get feedback on choice 
            R1 = 1 if target_feature[stage] in stim_hist[t][0] else -1
            R2 = 0 # reward for other stimulus is inferred to be the opposite
            reward.append([R1,R2])

            feedback.append(reward[t][stim_hist[t].index(choice[t])])
            pe.append(reward[t][stim_hist[t].index(choice[t])] - val[t][stim_hist[t].index(choice[t])]) # calculate prediction error

            # Update Weights ================
            w_a += alpha * (reward[t][0] - val[t][0]) * inputs[t][0][0].T
            w_b += alpha * (reward[t][0] - val[t][0]) * inputs[t][0][1].T
            w_a += alpha * (reward[t][1] - val[t][1]) * inputs[t][1][0].T
            w_b += alpha * (reward[t][1] - val[t][1]) * inputs[t][1][1].T

            weights_a.append(w_a.copy())
            weights_b.append(w_b.copy())

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
                           'weights_b': weights_b, 'alpha': alpha, 'beta': beta, 'weights_0': weights_0,\
                            'dimension1': dimension1, 'subject': subject, 'stage':list(itertools.chain(*[[stages[x]]*n_trials[x] for x in range(len(n_trials))])),\
                            'n_trials': n_trials, 'n_erros' :n_errors, 'stages_attempted' :stages_attempted, 'stages_passed': stages_passed}

                return results
            
            t+=1
    results = {'pe':pe, 'choice': choice, 'val': val, 'feature_weights': feature_weights, 'reward': reward, 'feedback': feedback, 'stim_hist': stim_hist, 'inputs': inputs, 'weights_a':weights_a,\
                'weights_b': weights_b, 'alpha': alpha, 'beta': beta, 'weights_0': weights_0,\
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
    from scipy.special import logsumexp

    nP = len(params)

    alpha, beta = params
    ab = np.asarray(params.copy()).reshape(nP,1)

    R,choice,S,dimension1,likelihood = args[:5]
    
    if not likelihood:
        # transform parameters
        alpha = 1/(1 + np.exp(-ab[0]))
        beta = np.exp(ab[1])

    if len(args) > 5:
        # we are doing EM
        u = args[5]
        v2 = args[6]
        u = u.reshape(nP,1)

        PP = ab - u
        L = 0.5*(PP.T @ np.linalg.pinv(v2) @ PP)
        LP = -np.log(2*np.pi) - 0.5*np.log(np.linalg.det(v2)) - L
        NLP = -LP[0] # calculate negative log liklihood of drawing these parameters from a given prior

    fs = ['A_01', 'A_02', 'A_03', 'A_04', 'A_05', 'A_06', 'A_07', 'A_08', 'A_09', 'A_10', 'A_11', 'A_12',
                'B_01', 'B_02', 'B_03', 'B_04', 'B_05', 'B_06', 'B_07', 'B_08', 'B_09', 'B_10', 'B_11', 'B_12']
    lines,shapes = dict(zip(fs[:12], np.identity(12))),dict(zip(fs[12:], np.identity(12)))

    AH,V,inputs = [],[],[]
    l,ll = [0]*2,0
    like = []
    
    wh0 = [np.array([0.0]*12).reshape(1,12),np.array([0.0]*12).reshape(1,12)]
    wh1 = wh0[0].copy()
    wh2 = wh0[1].copy()

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

            l = beta * (V[t][S[t].index(choice[t])] - vmax) - logsumexp([beta * (V[t][x] - vmax) for x in range(len(S[t]))])
            #l = beta*(V[t][S[t].index(choice[t])] - vmax) - np.log(sum((np.exp(beta*(V[t][x]-vmax)) for x in range(len(S[t])))))
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
            V1,V2 = np.dot(wh1, l1) + np.dot(wh2, s1),np.dot(wh1, l2) + np.dot(wh2, s2)
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

            wh1 += alpha * (R[t][0] - V[t][0]) * inputs[t][0][0].T
            wh2 += alpha * (R[t][0] - V[t][0]) * inputs[t][0][1].T
            wh1 += alpha * (R[t][1] - V[t][1]) * inputs[t][1][0].T
            wh2 += alpha * (R[t][1] - V[t][1]) * inputs[t][1][1].T

    if likelihood == True:
        return (np.prod(like), np.mean(like), like, -ll)


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
    
    warnings.simplefilter('ignore', RuntimeWarning)

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
                xopt = scipy.optimize.minimize(fun=llIED, x0=np.ravel(list(m)),args =  (args))
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
        return (params)


# In[ ]:




