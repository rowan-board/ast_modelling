#!/usr/bin/env python
# coding: utf-8

# In[1]:


def plot_idata(data):
                
    """Generate a series of plots given an individuals model simulated IED data. 

    Parameters:
    data (dict): dict containing simulated data on IED task.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # get number of trials in each stage
    trials = [sum(data['Trials'][:x]) for x in range(len(data['Trials']))]

    font = {'family' : 'Arial',
            'size'   : 22}

    matplotlib.rc('font', **font)
    
    # plot trial by trial choices
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    sns.scatterplot(list(range(len(data['choice']))),data['choice'],alpha = 0.9,s = 50)
    for i in trials:
        plt.axvline(i,color = 'k',alpha = 0.5)
    plt.xlabel('Trial')
    plt.show()
    
    # plot values of lines over trials
    colors = ['lightseagreen','midnightblue','grey','pink','slateblue','peru']
    labels = ['L0_6','L0_7','L0_8','L0_9','L0_10','L0_11']
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    for x in range(6):
        plt.plot((np.concatenate(data['WH1']))[:,x],color = colors[x],label = labels[x])
    plt.legend(fontsize = 10,frameon = False)
    for i in trials:
        plt.axvline(i,color = 'k',alpha = 0.5)
    plt.ylabel('Line Feature Weight')
    plt.xlabel('Trial')
    plt.show()

    # plot values of shapes over trials
    labels = ['S0_0','S0_1','S0_2','S0_3','S0_4','S0_5']
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    for x in range(6):
        plt.plot((np.concatenate(data['WH2']))[:,x],color = colors[x],label = labels[x])
    plt.legend(fontsize = 10,frameon = False)
    for i in trials:
        plt.axvline(i,color = 'k',alpha = 0.5)
    plt.ylabel('Shape Feature Weight')
    plt.xlabel('Trial')
    plt.show()
    
    if 'THE' in data:
        # plot dimension attention over trials
        fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
        the = list(np.concatenate(np.concatenate(data['THE'])))
        sns.lineplot(list(range(len(the))),the,color = 'darkcyan',alpha = 0.9)
        plt.xlabel('Trial')
        plt.ylabel('Dimension Weight')
        for i in trials:
            plt.axvline(i,color = 'k',alpha = 0.5)
        plt.axhline(y = 0, color = 'k', alpha=0.7, zorder=0,linestyle = ':')
        plt.show()
    else:
        pass

    # plot prediction error over trials
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    pe = list(np.concatenate(data['PE']))
    sns.lineplot(list(range(len(pe))),pe,color = 'midnightblue',alpha = 0.9)
    plt.ylabel('Prediction Error')
    for i in trials:
        plt.axvline(i,color = 'k',alpha = 0.5)
    plt.xlabel('Trial')
    plt.axhline(y = 0, color = 'k', alpha=0.7, zorder=0,linestyle = ':')
    plt.show()

    # plot errors per stage
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    sns.barplot(x = data['Stages'],y = data['Errors'],color = 'lightseagreen',alpha = 0.9)
    plt.ylabel('Errors')
    plt.xticks(rotation = 90)
    plt.show()

    # plot trials per stage
    fig,ax=plt.subplots(figsize=(15,8), dpi= 300, facecolor='w', edgecolor='k')
    sns.barplot(x = data['Stages'],y = data['Trials'],color = 'lightseagreen',alpha = 0.9)
    plt.ylabel('Trials to Criterion')
    plt.xticks(rotation = 90)
    plt.show()
    return

def load_trialdata(csv,subjects = None):
    
    """Load trial-by-trial data from csv. 

    Parameters:
    csv (dir): directory of csv containing trial-by-trial IED data.
    subjects (list): list of subjects to filter dataframe by (determined by e.g. data cleaning).
    
    Returns:
    dataframe: cleaned csv as dataframe.
   """
    
    import pandas as pd
    import pickle
    import numpy as np

    df1 = pd.read_csv(csv) # load in csv
    if subjects is not None:
        df1 = df1[df1.subject.isin(subjects)] # filter by list of subjects
    df1 = df1[~df1.duplicated(subset=None, keep='first')] # remove duplicate rows
    df1[['trials_IsAnswerCorrect']] = df1[['trials_IsAnswerCorrect']].astype(int).replace({0: -1}) # replace wrong answer as -1

    return df1

def trials2dict(dftrials,dimension1):
    
    """Convert trial-by-trial dataframe to list of dicts. 

    Parameters:
    dftrials (dataframe): dataframe of trial-by-trial IED data.
    dimension1 (str): The first relevant dimension - 'lines' or 'shapes'.
    
    Returns:
    list: list containing subject dicts with relevant data for fit functions.
   """
    
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    dicts = []
    subjects = dftrials.subject.unique() # get list of subjects from dataframe
    for i in tqdm(subjects,desc = 'Participant'):
        dftemp = dftrials[dftrials.subject == i].reset_index(drop=True) # filter by each participants data
        choice = [dftemp['trials_Box' + str(dftemp['trials_ChosenBox'].iloc[x])].iloc[x] for x in range(len(dftemp))] #get list of their choices

        S=[]
        for y in range(len(dftemp)):
            S.append([x for x in dftemp.iloc[y,5:9].tolist() if x is not np.nan].copy()) # get stimuli they saw on each trial

        R = []
        for index, row in dftemp.iterrows():
            r = [0]*2
            r[(S[index].index(choice[index]))] = row['trials_IsAnswerCorrect'] #assign chosen index for rewards this trial to outcome
            r[r.index(0)] = 0 - r[1-r.index(0)] # assign other index of rewards this trial to opposite
            R.append(r.copy()) #get outcomes associated with BOTH stimuli (counterfactual learning)
            
    
        dict1 = {'R':R, 'choice':choice, 'S':S, 'dimension1':dimension1}
        dicts.append(dict1)
    return(dicts)

def load_stagedata(csv,subjects):
    """Load summary stage dataframe. 

    Parameters:
    csv (dir): directory of csv containing stage level IED data.
    subjects (list): list of subjects to filter dataframe by (determined by e.g. data cleaning).
    
    Returns:
    dataframe: cleaned csv as pandas dataframe.
   """
    import pandas as pd
    realstages = pd.read_csv(csv) # load csv
    real1 = realstages.copy()
    real1 = real1[~real1.duplicated(subset=None, keep='first')] # remove duplicate rows
    real1 = real1[real1['subject'].isin(subjects)] # filter by subjects
    real2 = real1[["stages_StageNumber", "stages_TotalErrors",'stages_StagePassed','subject']] # filter by relevant columns
    real3 = real2.replace({'stages_StageNumber':{5:'Simple Discrimination',6:'Simple Reversal',7:'Compound Discrimination',                                                 8: 'Compound Discrimination 2',9:'Compound Reversal',10:'Intra-Dimensional Shift',                                             11:'Intra-Dimensional Reversal',12:'Extra-Dimensional Shift',13:'Extra-Dimensional Reversal'}}) # replace stage names
    real3['Data'] = 'Human' #add extra column for datatype

    return real3

def dicts2stages(dicts,subjects):  
    """Load summary stage dataframe. 

    Parameters:
    dicts (list): list containing subject data in type dict.
    subjects (list): list of subjects to filter dataframe by (determined by e.g. data cleaning).
    
    Returns:
    dataframe: dataframe of simulated data by stage.
   """
    import pandas as pd
    from tqdm import tqdm

    S = []
    for i in tqdm(range(len(dicts)),desc = 'Participant'): # create dataframe for each subject
        columns = ['stages_StageNumber','stages_TotalErrors','stages_StagePassed','Data','subject','alpha']
        for j in ['theta0','beta','epsilon']:
            if j in dicts[i]:
                columns.append(j)
                               
        stages = pd.DataFrame(columns = columns)
        stages['stages_StageNumber'] = dicts[i]['Stages']
        stages['stages_TotalErrors'] = dicts[i]['Errors']
        stages['stages_StagePassed'] = dicts[i]['Passed']
        stages['Data'] = 'Model'
        stages['subject'] = subjects[i]
        stages['alpha'] = dicts[i]['alpha']
        if 'theta0' in dicts[i]:
            stages['theta0'] = dicts[i]['theta0']
        if 'beta' in dicts[i]:
            stages['beta'] = dicts[i]['beta']
        if 'epsilon' in dicts[i]:
            stages['epsilon'] = dicts[i]['epsilon']
            
        S.append(stages) # append to list of dataframes
    Sall = pd.concat(S) # concat all dataframes
    return(Sall)

def assess_fit(trials_csv, stages_csv, subjects, dimension1, nP, fit_func, sim_func, transforms, n_jobs, names, like_func, seed = 42, fit = 'EM', rng = None, Nsamples = 5000, EM_iter = 100):
    """Master Function. Loads data, fits the model, simulates data and compares it with real data, qualitative and quantitative model fits.

    Parameters:
    trials_csv (dir): direcftory of csv containing trial-by-trial IED data.
    stages_csv (dir): directory of csv containing stage level IED data.
    subjects (dir): directory of pickled list of subjects to filter dataframe by (determined by e.g. data cleaning).
    dimension1 (str): The first relevant stimulus dimension - 'lines' or 'shapes'.
    nP (int): number of parameters to be estimated per subject.
    fit_func (func): function that fits data with required model.
    sim_func (func): function that simulates data from model.
    transforms (list): list of parameter transformations should be 'exp' for exponential, 'sigmoid' for sigmoid, or None for no transformation. Must be given if dist is given. Default = None.
    rng (func): pass an existing random number generator. Default = None.
    seed (int): if rng is None, this seed will be used to create a random number generator. Default = 42.
    n_jobs (int): number of processes to use for fitting function. Default = 1.
    fit (str): 'ML', or 'EM' depending on whether maximum likelihood or expectation maximisation should be used. Default = 'EM'.
    EM_iter: number of iterations of EM algorithm to complete. Default = 100.
    names (list): list of parameter names as strings to label recovery graphs.
    like_func (func): function to calculate likelihood with particular model.
    Nsamples (int): number of samples to draw from prior distribution to calculate iBIC. Default = 5000. 
    
    Returns:
    dict: keys are
    'est' - output from parameter estimation function
    'df_all' - dataframe of simulated and real data combined
    'iBIC' - output from iBIC function
   """                                  
    import modelling
    from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import pickle
    import numpy as np
                                               
    rng = rng if rng else np.random.default_rng(seed)

    with open (subjects, 'rb') as fp:
        subjects = pickle.load(fp)
                                               
    model_args = dimension1                                         

    # load trial by trial data
    print('**Loading Trial by Trial Data**')
    df = load_trialdata(trials_csv,subjects)
    # convert to classes for fit function
    data = trials2dict(df,dimension1)
    # find best fitting parameters for participants
    print('**Fitting Data**')
    est = modelling.fit(data=data,nP=nP,fit_args=['R','choice','S','dimension1'],fit_func=fit_func,transforms = transforms, rng = rng,              n_jobs= n_jobs,fit = fit,EM_iter = EM_iter)
    print('**Plotting Estimates**')
    for p in range(est['m'].shape[1]):
        fig=plt.figure(figsize=(8,5), dpi= 80, facecolor='w', edgecolor='k')
        ax=sns.violinplot(x = est['m'][:,p], color = '1',                              bw=.2,  linewidth=1,cut=0.,                           scale="area", width=.6, inner=None)
        ax=sns.stripplot(x = est['m'][:,p], color = '#008B8B',                         edgecolor="white",size=4,jitter=1,zorder=1, alpha=0.6)
        ax=sns.boxplot(x = est['m'][:,p], color = 'gray',                      width=.15, showcaps=True,boxprops={'facecolor':'none', "zorder":5},fliersize = 2,                       showfliers=True,whiskerprops={'linewidth':1, "zorder":5},saturation=0.6,linewidth=1 )
        name = names[p] if names else 'parameter ' + str(p+1)
        plt.title(name)
        plt.show()

    # simulate data with fit parameters
    print('**Simulating Data**')
    data2 = modelling.simulate(model = sim_func, transforms = transforms,model_args = model_args, rng = rng, params = est['m'],N = est['m'].shape[0])
    # convert simulated trials to stages
    simstages = dicts2stages(data2['data'],subjects)
    # load real stages data
    print('**Loading Stage Data**')
    stages = load_stagedata(stages_csv,subjects)
    print('**Assessing Fit**')
    df_all = pd.concat([stages, simstages])
    dftrue = df_all[df_all.stages_StagePassed == True]
    dffalse = df_all[df_all.stages_StagePassed == False]
    sns.set_style('ticks')
    fig=plt.figure(figsize=(9,5), dpi= 300)
    axes = plt.gca()
    axes.set_ylim([-2.5,39.5])
    ax = sns.violinplot(x = 'stages_StageNumber', y = 'stages_TotalErrors',data = df_all,hue = 'Data',                        palette = {'Human': '#5CB7B0', 'Model': '#4456A2'},inner = None,linewidth = 0,                        cut = 0,alpha = 0.3,split = True,width = 0.9,scale = 'count',scale_hue = False,bw =0.5)
    for violin, alpha in zip(ax.collections[::1],[0.3]*18):
        violin.set_alpha(alpha)

    ax = sns.stripplot(x = 'stages_StageNumber', y = 'stages_TotalErrors',data = dftrue, hue = 'Data',                        palette = {'Human': '#5CB7B0', 'Model': '#4456A2'},jitter = 0.3,dodge = True,                    size=4, edgecolor="gray", alpha=.4,order=["Simple Discrimination", "Simple Reversal", "Compound Discrimination",                    'Compound Discrimination 2', 'Compound Reversal','Intra-Dimensional Shift','Intra-Dimensional Reversal',                                    'Extra-Dimensional Shift','Extra-Dimensional Reversal'])
    ax = sns.stripplot(x = 'stages_StageNumber', y = 'stages_TotalErrors',data = dffalse, hue = 'Data',marker = 'X',                        palette = {'Human': '#5CB7B0', 'Model': '#4456A2'},jitter = 0.3,dodge = True,                    size=6, edgecolor="gray", alpha=.4,order=["Simple Discrimination", "Simple Reversal", "Compound Discrimination",                    'Compound Discrimination 2', 'Compound Reversal','Intra-Dimensional Shift','Intra-Dimensional Reversal',                                    'Extra-Dimensional Shift','Extra-Dimensional Reversal'])

    plt.ylabel('Errors',fontsize = 16,labelpad = 10)
    plt.xlabel('Stage',fontsize = 16,labelpad = 10)
    plt.xticks(fontsize = 16,rotation = 0)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8],['1','2','3','4','5','6','7','8','9'],fontsize = 16)
    plt.yticks(fontsize = 16)
    p1, = plt.plot(60, marker = "X", color = '#4456A2', markersize=5,alpha = .8,markeredgecolor = 'None',linestyle = 'None')
    p2, = plt.plot(61, marker = "X", color = '#5CB7B0', markersize=5,alpha = .8,markeredgecolor = 'None',linestyle = 'None')
    p3, = plt.plot(62, marker = "o", color = '#4456A2', markersize=3,alpha = .8,markeredgecolor = 'None',linestyle = 'None')
    p4, = plt.plot(63, marker = "o", color = '#5CB7B0', markersize=3,alpha = .8,markeredgecolor = 'None',linestyle = 'None')
    #plt.legend(handles = [p4,p2,p3,p1],labels = ['Human, Stage Passed','Human, Stage Failed', 'Naive RL, Stage Passed','Naive RL, Stage Failed'],\
       #       fontsize = 20, markerscale = 1.4,bbox_to_anchor=(1, 1.02),edgecolor = 'inherit')
    plt.legend([(p4, p2),(p3,p1)], ['Human: Passed, Failed','Model: Passed, Failed'], numpoints=1,
                   handler_map={tuple: HandlerTuple(ndivide=None)},fontsize = 11, markerscale = 1.5,bbox_to_anchor=(0.33, 1),edgecolor = 'w')
    #plt.legend(labels = ['Human','Naive RL'],)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    plt.show()

    # scatter plot of human vs model errors on ED shift
    temp = df_all[['subject','stages_StageNumber','stages_TotalErrors','Data']]
    temp2 = temp[temp['stages_StageNumber'] == 'Extra-Dimensional Shift']
    temp2 = temp2[['subject','stages_TotalErrors','Data']]
    original_df = temp2.pivot(index='subject', columns='Data')
    original_df.columns = original_df.columns.droplevel().rename(None)
    original_df = original_df.reset_index()
    original_df['Human'] = original_df['Human'] + np.random.normal(0,0.15,len(original_df))
    original_df['Model'] = original_df['Model'] + np.random.normal(0,0.15,len(original_df))

    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(5,5), dpi= 300, facecolor='w', edgecolor='k')
    sns.set_style('ticks')

    axes = plt.gca()
    axes.set_ylim([-2.5,37.5])
    axes.set_xlim([-2.5,37.5])
    ax = sns.scatterplot(x = 'Human',y = 'Model',data = original_df,color = '#5CB7B0',alpha = .5,s = 30,linewidth = 0,zorder = 1)
    ax = plt.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3",alpha = 0.2)
    plt.xlabel('Human',fontsize = 16,labelpad = 10)
    plt.xticks(fontsize = 16)
    plt.ylabel('Model',fontsize = 15,labelpad = 10)
    plt.yticks(fontsize = 16)

    plt.show()

    print('**Calculating Average ED Shift Correlations**')

    c=[]
    while len(c) < 10:
        # simulate with fit parameters
        data3 = modelling.simulate(model = sim_func, transforms = transforms,model_args = model_args, rng= rng, params = est['m'],N = est['m'].shape[0])
        # convert trials to stages
        simstages2 = dicts2stages(data3['data'],subjects)
        # concatenate
        df_all2 = pd.concat([stages, simstages2])
        temp = df_all2[['subject','stages_StageNumber','stages_TotalErrors','Data']]
        temp2 = temp[temp['stages_StageNumber'] == 'Extra-Dimensional Shift']
        temp2 = temp2[['subject','stages_TotalErrors','Data']]
        original_df = temp2.pivot(index='subject', columns='Data')
        original_df.columns = original_df.columns.droplevel().rename(None)
        original_df.reset_index()
        c.append(original_df.corr().iloc[0,1])
    print('Average Correlation: ',np.mean(c))

    print('**Calculating iBIC**')
    iBIC = modelling.iBIC(u = est['u'],v2 = est['v2'],data= data, transforms = transforms, like_func = like_func,fit_args = ['R','choice','S','dimension1'], rng = rng, n_jobs = n_jobs, Nsamples = Nsamples)

    print('iBIC: ', iBIC['iBIC'])

    print('**Calculating Average Likelihood per Trial**')
    alpt = modelling.alpt(params = est['m'],data = data, like_func = like_func,fit_args = ['R','choice','S','dimension1'],                  n_jobs = n_jobs)

    print('Average Likelihood per Trial: ', alpt['alpt'])
                                               

    results = {'est':est, 'df_all': df_all,'iBIC': iBIC}
    return results


