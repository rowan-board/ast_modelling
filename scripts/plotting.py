# IED script for plotting the behavioural and modelling data

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def group_level_plots(data, groups, blocks):

    # mouse data with are GH, ISO, and DREADDs groups
    if len(groups) == 4:
        group_1 = groups[:2]
        group_2 = groups[2:]
        group_list = [group_1, group_2]

        for group in group_list:
            tmp_df = data[['C', 'reward_loc','block', 'mouse', 'condition']].copy()
            tmp_df['correct'] = (tmp_df['C'] == tmp_df['reward_loc']).astype(int)
            tmp_df = tmp_df[tmp_df['condition'].isin(group)]
            tmp_df = tmp_df[tmp_df['block'].isin(blocks)]

            # get number of trials to criterion for each mouse for each block
            t_block = tmp_df.groupby(['mouse', 'block'])['block'].value_counts().reset_index(name='count')
            errors = tmp_df.groupby(['mouse', 'block']).sum('C').reset_index()
            errors = t_block['count'] - errors['correct']
            t_block['errors'] = errors

            # get condition for each mouse and join to t_block 
            condition = tmp_df[['mouse', 'condition']].drop_duplicates()
            t_block = pd.merge(t_block, condition, on='mouse')

            # make sure blocks are in correct order for plot
            t_block['block'] = pd.Categorical(t_block['block'], categories=['SD', 'CD', 'IDS1', 'IDS2', 'IDS3', 'IDS4', 'EDS', 'EDSR'])

            # line plot
            ax = sns.lineplot(data=t_block, x = 'block', y = 'errors', hue='condition', errorbar = 'se')
            ax.set_ylim(0,25)
            ax.set_title('Errors per stage, ' + group[0] + ' vs ' + group[1])

            plt.show()
    else:
        # just plot the specified groups
        tmp_df = data[['C', 'reward_loc','block', 'mouse', 'condition']].copy()
        tmp_df['correct'] = (tmp_df['C'] == tmp_df['reward_loc']).astype(int)
        tmp_df = tmp_df[tmp_df['condition'].isin([groups])]
        tmp_df = tmp_df[tmp_df['block'].isin([blocks])]

        # get number of trials to criterion for each mouse for each block
        t_block = tmp_df.groupby(['mouse', 'block'])['block'].value_counts().reset_index(name='count')
        errors = tmp_df.groupby(['mouse', 'block']).sum('C').reset_index()
        errors = t_block['count'] - errors['correct']
        t_block['errors'] = errors

        

        # get condition for each mouse and join to t_block 
        condition = tmp_df[['mouse', 'condition']].drop_duplicates()
        t_block = pd.merge(t_block, condition, on='mouse')

        # make sure blocks are in correct order for plot
        t_block['block'] = pd.Categorical(t_block['block'], categories=['SD', 'CD', 'IDS1', 'IDS2', 'IDS3', 'IDS4', 'EDS', 'EDSR'])

        # line plot
        ax = sns.lineplot(data=t_block, x = 'block', y = 'errors', hue='condition', errorbar = 'se')
        ax.set_ylim(0,20)
        ax.set_title('Errors per stage, group housed vs isolated')

    
    # now lets plot shift cost bars

    # reset the data with the groups we want
    tmp_df = data[['C', 'reward_loc','block', 'mouse', 'condition']].copy()
    tmp_df['correct'] = (tmp_df['C'] == tmp_df['reward_loc']).astype(int)
    tmp_df = tmp_df[tmp_df['condition'].isin(groups)]
    tmp_df = tmp_df[tmp_df['block'].isin(blocks)]

    # get number of trials to criterion for each mouse for each block
    t_block = tmp_df.groupby(['mouse', 'block'])['block'].value_counts().reset_index(name='count')
    errors = tmp_df.groupby(['mouse', 'block']).sum('C').reset_index()
    errors = t_block['count'] - errors['correct']
    t_block['errors'] = errors

    # get condition for each mouse and join to t_block 
    condition = tmp_df[['mouse', 'condition']].drop_duplicates()
    t_block = pd.merge(t_block, condition, on='mouse')

    # count number of trials at IDS4 and EDS for each subj
    IDS4 = t_block[(t_block['block'] == 'IDS4')]
    EDS =  t_block[(t_block['block'] == 'EDS')]


    shift_costs = pd.merge(IDS4, EDS, on = 'mouse')
    shift_costs['costs'] = shift_costs['errors_y'] - shift_costs['errors_x']

    custom_palette = {'GH': sns.color_palette("Paired")[5], 'ISO': sns.color_palette("Paired")[1],
                       'GH DREADDs': sns.color_palette("Paired")[4], 'ISO DREADDs': sns.color_palette("Paired")[0]}
    plt.figure(figsize=(3,5))
    ax = sns.boxplot(shift_costs, x = 'condition_x', y = 'costs', order=groups, hue='condition_x', palette=custom_palette)
    ax = sns.stripplot(shift_costs, x = 'condition_x', y = 'costs', order=groups, hue='condition_x', palette=custom_palette, edgecolor='k', linewidth=1)
    ax.set_title('Shift costs for each group')
    ax.set_xlabel('Condition')
    ax.set_ylim(-7.5,12.5)
    plt.savefig('plots/shift_costs_controls.png')
    plt.show
    return shift_costs


def individual_plots(data, groups, blocks):

    # first let's get all the shift costs for all mice so we can plot the distriubtion
    # reset the data with the groups we want
    tmp_df = data[['C', 'reward_loc','block', 'mouse', 'condition']].copy()
    tmp_df['correct'] = (tmp_df['C'] == tmp_df['reward_loc']).astype(int)
    tmp_df = tmp_df[tmp_df['block'].isin(blocks)]

    # get number of trials to criterion for each mouse for each block
    t_block = tmp_df.groupby(['mouse', 'block'])['block'].value_counts().reset_index(name='count')
    errors = tmp_df.groupby(['mouse', 'block']).sum('C').reset_index()
    errors = t_block['count'] - errors['correct']
    t_block['errors'] = errors

    # get condition for each mouse and join to t_block 
    condition = tmp_df[['mouse', 'condition']].drop_duplicates()
    t_block = pd.merge(t_block, condition, on='mouse')

    # count number of trials at IDS4 and EDS for each subj
    IDS4 = t_block[(t_block['block'] == 'IDS4')]
    EDS =  t_block[(t_block['block'] == 'EDS')]

    shift_costs = pd.merge(IDS4, EDS, on = 'mouse')
    shift_costs['costs'] = shift_costs['errors_y'] - shift_costs['errors_x']

    all_costs = shift_costs['costs']

    # reset the data with the groups we want
    tmp_df = data[['C', 'reward_loc','block', 'mouse', 'condition']].copy()
    tmp_df['correct'] = (tmp_df['C'] == tmp_df['reward_loc']).astype(int)
    tmp_df = tmp_df[tmp_df['condition'].isin(groups)]
    tmp_df = tmp_df[tmp_df['block'].isin(blocks)]

    # get number of trials to criterion for each mouse for each block
    t_block = tmp_df.groupby(['mouse', 'block'])['block'].value_counts().reset_index(name='count')
    errors = tmp_df.groupby(['mouse', 'block']).sum('C').reset_index()
    errors = t_block['count'] - errors['correct']
    t_block['errors'] = errors
    # make sure blocks are in correct order for plot
    t_block['block'] = pd.Categorical(t_block['block'], categories=['SD', 'CD', 'IDS1', 'IDS2', 'IDS3', 'IDS4', 'EDS', 'EDSR'])
    n = len(t_block['mouse'].unique()) 

    fig, axs = plt.subplots(n,2)

    fig.set_figheight(2.5*n)
    fig.set_figwidth(2*n)

    # loop over mice
    i=0
    for mouse in t_block['mouse'].unique():

        # get the individual subj we want to plot
        tmp_df = t_block[t_block['mouse'] == mouse]

        # let's plot errors over blocks
        sns.lineplot(tmp_df,x='block',y='errors',ax = axs[i,0])
        condition = data[data['mouse'] == mouse]['condition'].unique() # get the condition for the plot title
        axs[i,0].set_title(mouse + ', ' + condition)

        # plot shift cost of current mouse on dist of all shift costs
        sns.histplot(all_costs,ax=axs[i,1])
        axs[i,1].set_xlim(-10,20)
        subj_shift_cost = shift_costs[shift_costs['mouse'] == mouse]['costs'].tolist()
        axs[i,1].axvline(x=subj_shift_cost, color='red') 

        i += 1

    plt.tight_layout()
    plt.show