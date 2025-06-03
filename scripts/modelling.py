#!/usr/bin/env python
# coding: utf-8

# In[13]:



def simulate(model, model_args = 'lines', rng = None, seed = 42, params=None, dist=None, N=100, transforms = None, subjects = None):
    """Simulate data from a given model and set of parameters (or drawn from a given dstirbution). 

    Parameters:
    model (func): function that simulates data from model.
    model_args (str): additional arguments to be passed to model function. For IED model this is either 'lines' or 'shapes' indicating the first relevant stimulus dimension. Default = 'lines'.
    rng (func): pass an existing random number generator. Default = None.
    seed (int): if rng is None, this seed will be used to create a random number generator. Default = 42.
    params (array): N x nP array of transformed parameter values to be used for simulation. Default = None.
    dist (tuple): tuple of (mean, covariance) arrays for drawing untransformed parameters from a multivariate gaussian distribution if params is none. Mean should be 1 x nP array, and covariance should be a nP x nP array. Parameters will be transformed by transforms before being used for simulation. Default = None.
    N (int): number of participants data to simulate if params is None. Default = 100.
    transforms (list): list of parameter transformations should be 'exp' for exponential, 'sigmoid' for sigmoid, or None for no transformation. Must be given if dist is given. Default = None.
    subjects (list): subject names to add to simulated data. Deafult = None.

    Returns:
    dict: keys are 'params' and 'data' - 'params' is an array of transformed parameters used for simulation, 'data' is a list containing all simulated data.

   """
    import numpy as np 
    
    rng = rng if rng else np.random.default_rng(seed)
    
    if (model is None): 
        raise Exception('Essential model argument not received.') 

    # if no parameters have been given, draw N parameter vectors from a Normal distribution with mean and 
    # covariance given by dist
    if params is None:
        if (dist is None) | (transforms is None): 
            raise Exception('dist or transforms argument not received. Both must be given if params is not given.') 
        params = rng.multivariate_normal(dist[0], dist[1], N)
        # transform parameters to be in correct ranges for simulation
        nP = params.shape[1]
        for x in range(nP):
            if transforms[x] == 'sigmoid':
                params[:,x] = 1/(1+np.exp(-params[:,x]))
            elif transforms[x] == 'exp':
                params[:,x] = np.exp(params[:,x])
    else:
        # set N data points to n rows of params array
        N = params.shape[0]
        
    # if no subject names are given, use integers from 0 to N-1
    subjects = subjects if subjects else list(range(N))

    # simulate N datapoints using provided model function, parameters and additional model arguments
    data = [model(params = params[x,:],dimension1 = model_args,subject = subjects[x],rng = rng) for x in range(N)]

    results = {'params': params, 'data': data}
    
    return results

def fit(data, nP, fit_func, fit_args, transforms, rng = None, seed = 42, n_jobs = 1, fit = 'EM', EM_iter = 100):
    
    """Estimate parameter values from data given a model. 

    Parameters:
    data (list): subjects data in the form of a list of dicts. 
    nP (int): number of parameters to be estimated per subject.
    fit_func (func): function that fits data with required model.
    fit_args (list): list of additional arguments to be passed to fit_func function.
    transforms (list): list of parameter transformations should be 'exp' for exponential, 'sigmoid' for sigmoid, or None for no transformation. Must be given if dist is given. Default = None.
    rng (func): pass an existing random number generator. Default = None.
    seed (int): if rng is None, this seed will be used to create a random number generator. Default = 42.
    n_jobs (int): number of processes to use for fitting function. Default = 1.
    fit (str): 'ML', or 'EM' depending on whether maximum likelihood or expectation maximisation should be used. Default = 'EM'.
    EM_iter: number of iterations of EM algorithm to complete. Default = 100.

    Returns:
    dict: keys are 'm', 's2', 'u', 'v2' - if fit is 'EM', 'm' is array of indivudals best fitting parameter estimates, 's2' is individual parameter covariances, 'u' is group level prior mean estimate, 'v2' is group level prior covariance estimate. If fit is 'ML', dict only contains 'm'.

   """
    import numpy as np
    import scipy.linalg
    from multiprocessing import Pool
    import tqdm
    import warnings

    #warnings.simplefilter('ignore', RuntimeWarning)

    rng = rng if rng else np.random.default_rng(seed)
    
    if (data is None) | (nP is None) | (fit_args is None) | (fit_func is None) | (transforms is None): 
        raise Exception('At least one of essential arguments data, nP, fit_args, fit_func, transforms not received.') 

    N = len(data) # set number of participants as length of data list
    v20 = 0.1*np.identity(nP)
    m = np.matmul(rng.standard_normal((N,nP)),(scipy.linalg.sqrtm(np.linalg.inv(v20)))) # initialise individual parameters

    if fit == 'ML':
    # maximum likelihood estimation
        # generate extra arguments required for fit function one participant at a time
        args = ([data[x][fit_args[y]] for y in range(len(fit_args))] + [False,m[x,:]] for x in range(N))
        p=Pool(n_jobs)

        # use multiprocessing to fit each participants parameters
        results = list(tqdm.tqdm(p.imap(fit_func,args),desc= 'Participant'))
        p.close()
        p.join()

        m = np.array(results)

        # transform parameters back to correct range values
        for x in range(nP):
            if transforms[x] == 'sigmoid':
                m[:,x] = 1/(1+np.exp(-m[:,x]))
            elif transforms[x] == 'exp':
                m[:,x] = np.exp(m[:,x])


        results = {'m':m}
        return results

    else:
        if (fit != 'EM'): 
            raise Exception('fit argument must be set to "ML" or "EM".') 
        u0 = np.mean(m,axis = 0) # set prior mean to mean of m's
        u,v2 = u0.copy(),v20.copy()
        U,V2 = [u.copy()], [v2.copy()]
        s2 = np.tile(v20.copy(),N) # set individual variances to prior variance
        M,S2 = [m.copy()],[s2.copy()]

        for t in range(EM_iter):

            # E STEP
            args = ([data[x][fit_args[y]] for y in range(len(fit_args))] + [False,u,v2,m[x,:],s2[:,x*nP:(x+1)*nP],rng] for x in range(N))
            p=Pool(n_jobs)
            results = list(tqdm.tqdm(p.imap(fit_func,args),desc= 'Participant'))
            p.close()
            p.join()
            for x in range(N):
                m[x,:] = results[x][0]['x'] # set participants m's to argmin(h) from minimisation
                s2[:,x*nP:(x+1)*nP] = results[x][0].hess_inv # set participants variance to inverse hessian          
            M.append(m.copy())
            S2.append(s2.copy())

            # M step
            u = np.mean(m,axis = 0) # set prior mean to mean of m's
            v2 = sum([np.outer(m[x,:],m[x,:]) + s2[:,x*nP:(x+1)*nP] for x in range(N)])/N - np.outer(u,u) # set prior variance 

            U.append(u.copy())
            V2.append(v2.copy())

            print('Iteration:',t + 1,u,v2) # print iteration number and  current prior parameter estimates

            # stop if prior means change by less than 0.01 and prior stds change by less than 0.1 between iterations
            if len(U) >= 15 and np.all(abs(np.subtract(U[-1],U[-2])) < 1e-3) and np.all(abs(np.subtract(V2[-1],V2[-2])) < 1e-3):
                break

        for x in range(nP):
            if transforms[x] == 'sigmoid':
                m[:,x] = 1/(1+np.exp(-m[:,x]))
            elif transforms[x] == 'exp':
                m[:,x] = np.exp(m[:,x])
                
                
        results = {'m':m,'s2':s2,'u':u, 'v2':v2}
 
        return results


def alpt(data, like_func, fit_args, params, n_jobs = 1):
    
    """Calculate average likelihood per trial given data, a model, and parameter values. 

    Parameters:
    data (list): subjects data in the form of a list of dictionaries. 
    like_func (func): function to calculate likelihood with particular model.
    fit_args (list): list of additional arguments to be passed to like_func function.
    params (array): N (n subjects) x nP (n parameters) array of transformed parameter values.
    n_jobs (int): number of processes to use when calculating average likelihoods. Default = 1.

    Returns:
    dict: keys are 'alpt' and 'subject_alpts' - 'alpt' is the average likelihood per trial over all subjects, 'subject_alpts' is a list of average likelihood per trials for each subject.
   """
    
    import numpy as np
    import tqdm
    from multiprocessing import Pool
    
    if (data is None) | (like_func is None) | (fit_args is None) | (params is None): 
        raise Exception('At least one of essential arguments data, like_func, fit_args, params not received.') 

    N = len(data) # set N datapoints to length of data list
    assert N == params.shape[0]
    args = ((params[i,:],[data[i][j] for j in fit_args] + [True]) for i in range(N)) # get required data to pass to function
    p=Pool(n_jobs)
    result = list(tqdm.tqdm(p.starmap(like_func, args),desc = 'Participant')) # calculate average likelihood per trial
    p.close()
    p.join()
    alpts = [result[x][1] for x in range(len(result))] # get list of alpts for each subj
    likes = [result[x][2] for x in range(len(result))] # list of trialwise likelihood for each subj

    # get theta and alpha_d if the model has them
    if len(result[0]) > 5:
        alpha_d = [result[x][5] for x in range(len(result))] # theta over trials for each subj
        theta = theta = [result[x][4] for x in range(len(result))] # theta over trials for each subj
        results = {'alpt':np.mean(alpts), 'subject_alpts':alpts, 'trialwise_likelihood': likes, 'theta': theta, 'alpha_d': alpha_d}
    else:
        results = {'alpt':np.mean(alpts), 'subject_alpts':alpts, 'trialwise_likelihood': likes}
    if len(result[0]) == 5:
        theta = theta = [result[x][4] for x in range(len(result))] # theta over trials for each subj
        results = {'alpt':np.mean(alpts), 'subject_alpts':alpts, 'trialwise_likelihood': likes, 'theta': theta}

    # get results
    return results

def BIC(data, like_func, fit_args, params, n_jobs = 1):
    """Calculate bayesian information criterion (BIC) given data, a model, and parameter values. 

    Parameters:
    data (list): subjects data in the form of a list of dictionaries. 
    like_func (func): function to calculate likelihood with particular model.
    fit_args (list): list of additional arguments to be passed to like_func function.
    params (array): N (n subjects) x nP (n parameters) array of transformed parameter values.
    n_jobs (int): number of processes to use when calculating average likelihoods. Default = 1.

    Returns:
    dict: keys are 'alpt' and 'subject_alpts' - 'alpt' is the average likelihood per trial over all subjects, 'subject_alpts' is a list of average likelihood per trials for each subject.
   """
    
    import numpy as np
    import tqdm
    from multiprocessing import Pool
    
    if (data is None) | (like_func is None) | (fit_args is None) | (params is None): 
        raise Exception('At least one of essential arguments data, like_func, fit_args, params not received.') 

    N = len(data) # set N datapoints to length of data list
    assert N == params.shape[0]
    args = ((params[i,:],[data[i][j] for j in fit_args] + [True]) for i in range(N)) # get required data to pass to function
    p=Pool(n_jobs)
    result = list(tqdm.tqdm(p.starmap(like_func, args),desc = 'Participant')) # calculate average likelihood per trial
    p.close()
    p.join()

    # calculate bic for each subj
    n_trials = [result[x][2] for x in range(len(result))] # get n trials for each participant by taking length of likelihoods of all choices
    n_trials = [len(n_trials[x]) for x in range(len(n_trials))]
    ll = [result[x][3] for x in range(len(result))] # get summed log likelihood for each subj
    n_params = len(params[0,:])
    bic = [n_params * np.log(n_trials[i]) - 2 * ll[i] for i in range(len(ll))]

    results = {'log_likelihood': ll, 'BIC': bic}
    return results
    
def iBIC(u, v2, data, transforms, like_func, fit_args, rng= None, seed = 42, Nsamples = 5000, n_jobs = 1):
    
    """Calculate integrated Bayesian Information Criterion given data, a model, and a group level prior distribution (mean and covariance). 

    Parameters:
    u (array): mean of group level prior distribution - 1 x nP array. 
    v2 (array): covariance of group level prior distribution - nP x nP array. 
    data (list): subjects data in the form of a list of dictionaries.
    transforms (list): list of parameter transformations should be 'exp' for exponential, 'sigmoid' for sigmoid, or None for no transformation. Must be given if dist is given. Default = None.
    like_func (func): function to calculate likelihood with particular model.
    fit_args (list): list of additional arguments to be passed to like_func function.
    rng (func): pass an existing random number generator. Default = None.
    Nsamples (int): number of samples to draw from prior distribution to calculate iBIC. Default = 5000.    
    n_jobs (int): number of processes to use when calculating likelihoods. Default = 1.

    Returns:
    dict: keys are 'iBIC', 'sublog' and 'n' - 'iBIC' is the overall integrated Bayesian Information Criterion value, 'sublog' is a list of subject level fits, 'n' is the total number of datapoints.
   """
    
    import numpy as np
    from tqdm import tqdm
    from multiprocessing import Pool
    
    rng = rng if rng else np.random.default_rng(seed)
    
    if (u is None) | (v2 is None) | (data is None) | (transforms is None) | (like_func is None) | (fit_args is None): 
        raise Exception('At least one of essential arguments u, v2, data, transforms, like_func, fit_args not received.') 
    
    samples = rng.multivariate_normal(u,v2,Nsamples) # draw N samples from a multivarite normal distribution with given parameters
    nP = v2.shape[0] # number of parameters set to shape of prior covariance
    N = len(data) # number of participants set to length of data list

    # transform parameters to suitable ranges
    for x in range(nP):
            if transforms[x] == 'sigmoid':
                samples[:,x] = 1/(1+np.exp(-samples[:,x]))
            elif transforms[x] == 'exp':
                samples[:,x] = np.exp(samples[:,x])

    sublog,n = [],0 # set values to be updates

    for sub in tqdm(range(N), desc = 'Participant'): # iterate over subjects

        n+=len(data[sub]['choice']) # add n data points for each subject

        args = ((samples[i,:],[data[sub][j] for j in fit_args] + [True]) for i in range(Nsamples))
        p=Pool(n_jobs)
        subl = p.starmap(like_func, args) # calculate likelihood given parameters and required arguments
        p.close()
        p.join()
        subl2 = [subl[x][0] for x in range(len(subl))]
        sublog.append(np.log(np.mean(subl2)))
    iBIC = sum(sublog) - ((nP)*np.log(n))
    

    results = {'iBIC': iBIC,'sublog': sublog, 'n': n}
    return results
    
def recovery(sim_params,fit_params,names = None,color = 'lightseagreen',alpha = 0.5,s = 80):
    
    """Assess and plot parameter recovery given simulated and estimated parameter values). 

    Parameters:
    sim_params (array): array of simulated parameters. 
    fit_params (array): array of estimated parameters. 
    names (list): list of parameter names as strings to label recovery graphs.
    color (str): colour to use for plotting points on recovery graphs.
    alpha (float): transparency to use for plotting points on recovery graphs.
    s (int): size to use for plotting points on recovery graphs.
    """
        
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy import stats
    import matplotlib
    
    if (sim_params is None) | (fit_params is None): 
        raise Exception('At least one of essential arguments sim_params, fit_params.') 

    font = {'size'   : 22}

    matplotlib.rc('font', **font)

    for x in range(sim_params.shape[1]):

        fig,ax=plt.subplots(figsize=(15,15), dpi= 300, facecolor='w', edgecolor='k')
        ax = sns.scatterplot(x = sim_params[:,x],y = fit_params[:,x],color = color,s = s, alpha = alpha,edgecolor = None)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
               ]
        # now plot both limits against eachother
        ax.plot(lims, lims, 'k:', alpha=0.7, zorder=0)
        name = names[x] if names else 'parameter ' + str(x+1)
        nas = np.logical_or(np.isnan(fit_params[:,x]), np.isinf(fit_params[:,x]))
        print('%s corr:%f, p:%f' % (name,stats.pearsonr(sim_params[:,x][~nas],fit_params[:,x][~nas])[0],                                        stats.pearsonr(sim_params[:,x][~nas],fit_params[:,x][~nas])[1]))
        plt.xlabel('Simulated')
        plt.ylabel('Recovered')
        plt.title(name)
        sns.despine()
        plt.show()


# In[5]:






