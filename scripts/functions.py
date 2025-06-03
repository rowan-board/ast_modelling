# IED script for preprocessing functions to clean up our scripts

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import re

######## function to make our datasets for the AST/IED task consistent
def consistent(data):

    # isolate columns we want
    good_cols = ['LeftStim', 'RightStim', 'RewardedStim', 'RewardedLocation', 'Animal', 'Correct?', 'Session', 'Block', 'condition']
    data = data[good_cols].copy()

    # rename columns
    data.rename(columns = {'correct?': 'correct'}, inplace=True) 

    # convert col names to snake_case for consistency
    for i in range(len(data.columns)):
        new_col = re.sub(r'(?<!^)(?=[A-Z])', '_', data.columns[i]).lower()
        data.rename(columns = {data.columns[i]: new_col}, inplace=True)

    return data

# function to clean the raw behavioural data (original written be alfredo i think)
def clean_ied(data):

    # make an empty data frame for the clean data
    clean_data = pd.DataFrame()

    # get a list of mice and blocks
    mice = data.animal.unique()
    blocks = data.block.unique()
    blocks = [b for b in blocks if b != 'NoRule']

    for mouse in mice: 

        for block in blocks: 

            df = pd.DataFrame(columns = ["C", "O", "M","reward", "reward_loc", "O1", "O2", "M1", "M2","relevant_dim", "block", "day", "session", "mouse", "condition"])
            temp = data.loc[(data.animal == mouse) & (data.block == block)]

            if mouse == 'AMA425' and block == '1_IDS1':
                tmp = temp.iloc[0]

            #Let's get reward location and reward down
            df['reward'] = temp['correct?']
            df.loc[temp['rewarded_location'] == 'Left', 'reward_loc'] = 1 #Reward on left side = 1
            df.loc[temp['rewarded_location'] == 'Right', 'reward_loc'] = 0 #Reward on right side = 0

            df.loc[df.reward == 1, 'C'] = df.loc[df.reward == 1].reward_loc #Where the animal was correct the choice was the rewarded location
            df.loc[df.reward == 0, 'C'] = 1 - df.loc[df.reward == 0].reward_loc #Where it was incorrect the choice is the opposite of the rewarded location (1- 1 or 0)

            #Get rew stimulus
            if temp.rewarded_stim.all(): #If all items in rewarded stimuli are equal (they should, assign0 to variable)

                rewarded_stim = temp.iloc[0]['rewarded_stim']

            else:
                raise Exception("Something went wrong, rewarded stimulus is not the same throughout block") 

            odour_list = ['Lemon', 'Rosemary', 'Vanilla', 'Clove', 'Lavender', 'Cinnamon',
                            'Citronella', 'Thyme', 'Anise', 'Ginger', 'Nutmeg', 'Tumeric']
            texture_list = ['WhBed', 'Pipes', 'Popurri', 'WhShred','Felt', 'Tissue', 'Cord',
                            'MetalStrip', 'Cotton', 'Paper', 'Card', 'Ribbon']
            

            if rewarded_stim in odour_list:
                relevant_dim = 'O'

                L = temp.iloc[0]['left_stim']
                L = L.split('/')

                R = temp.iloc[0]['right_stim']
                R = R.split('/')

                [O1, O2] = [L[0], R[0]] #Odour 1 and 2
                [M1, M2] = [L[1], R[1]] #Medium 1 and 2

            elif rewarded_stim in texture_list: 
                relevant_dim = 'M'
                L = temp.iloc[0]['left_stim']
                L = L.split('/')

                R = temp.iloc[0]['right_stim']
                R = R.split('/')

                [O1, O2] = [L[1], R[1]] #Odour 1 and 2
                [M1, M2] = [L[0], R[0]] #Medium 1 and 2
        

            df.relevant_dim = relevant_dim
            df.O1 = O1 #Assign which stimulus is which to dataframe
            df.O2 = O2
            df.M1 = M1
            df.M2 = M2

            for x in range(2): #for 0 (right) and 1 (left)
                #then first feature in chosen stimulus is odour, second is media
                if x == 0: 
                    stim_list = [stim.split('/') for stim in temp.loc[df['C'] == x, 'right_stim']] 
                else: 
                    stim_list = [stim.split('/') for stim in temp.loc[df['C'] == x, 'left_stim']]

                O_values = [0 if O1 in stim else 1 for stim in stim_list] #0 if O1 or M1 chosen, 1 if M2 or O2 chosen
                M_values = [0 if M1 in stim else 1 for stim in stim_list]
                df.loc[df['C'] == x, 'O'] = O_values
                df.loc[df['C'] == x, 'M'] = M_values

            # Now give last couple variable columns
            df.session = temp.iloc[0].session
            df.mouse = mouse

            # split the block by name by the day, treating 1_IDS 2_IDS as one day for model, add extra 'day' col
            df.day = block.split('_')[0]
            df.day = df.day.replace('SD', '1')
            df.block = block.split('_')[-1]
            df.condition = temp.iloc[0]['condition']

            clean_data = pd.concat([clean_data, df])

    return clean_data


######### function to change all features names to A01, A02, A03, B01, B02, B03... so consistent across all mice
def standardize_features(data):

    # get a list of mice to iterate over
    mice = data.mouse.unique()

    ## first if stimuli match (i.e in simple discrim) then change to 'match' 
    # we won't use these features in the model

    # Process odours
    mask_odours = data['O1'] == data['O2']
    data.loc[mask_odours, ['O1', 'O2']] = 'match'

    # Process textures
    mask_textures = data['M1'] == data['M2']
    data.loc[mask_textures, ['M1', 'M2']] = 'match'

    all_O1s = []
    all_O2s = []
    all_T1s = []
    all_T2s = []

    for mouse in mice:

        tmp_data = data.loc[data['mouse'] == mouse ].reset_index(drop=True).copy()

        odours = []
        textures = []

        # loop over dataset
        for index, row in tmp_data.iterrows():
            
            if row['O1'] == 'match' or row['O2'] == 'match' or row['M1'] == 'match' or row['M2'] == 'match':
                continue

            # get list of odours in order that they appear
            if row['O1'] not in odours:
                odours.append(row['O1'])
            elif row['O2'] not in odours:
                odours.append(row['O2'])

            # get list of odours in order that they appear
            if row['M1'] not in textures:
                textures.append(row['M1'])
            elif row['M2'] not in textures:
                textures.append(row['M2'])

        # now create list of odours and textures to use for new col
        odours_i = ['A_01', 'A_02', 'A_03', 'A_04', 'A_05', 'A_06', 'A_07', 'A_08', 'A_09', 'A_10', 'A_11', 'A_12']
        textures_i = ['B_01', 'B_02', 'B_03', 'B_04', 'B_05', 'B_06', 'B_07', 'B_08', 'B_09', 'B_10', 'B_11', 'B_12']

        tmp_o1 = tmp_data['O1'].copy()
        tmp_o2 = tmp_data['O2'].copy()
        tmp_t1 = tmp_data['M1'].copy()
        tmp_t2 = tmp_data['M2'].copy()

        for i in range(len(tmp_o1)):
            if tmp_o1[i] == 'match':
                tmp_o1[i] = 'match'
            else:
                tmp_o1[i] = odours_i[odours.index(tmp_o1[i])]
            if tmp_o2[i] == 'match':
                tmp_o2[i] == 'match'
            else:
                tmp_o2[i] = odours_i[odours.index(tmp_o2[i])]
            if tmp_t1[i] == 'match':
                tmp_t1[i] = 'match'
            else:
                tmp_t1[i] = textures_i[textures.index(tmp_t1[i])]
            if tmp_t2[i] == 'match':
                tmp_t2[i] = 'match'
            else:
                tmp_t2[i] = textures_i[textures.index(tmp_t2[i])]

        all_O1s.extend(tmp_o1)
        all_O2s.extend(tmp_o2)
        all_T1s.extend(tmp_t1)
        all_T2s.extend(tmp_t2)

    # now append as new columns to clean_data
    data['O1_i'] = all_O1s
    data['O2_i'] = all_O2s
    data['T1_i'] = all_T1s
    data['T2_i'] = all_T2s

    return data


######### get the data in a format the data the model accepts
# now lets get the data in the correct format for the model 
# condense each participant down to 1 row, with vector in each for all data

# basically we need 4 variables
# R - reward vector of rewards, contains reward value for each option for each trial e.g [1, -1] if reward on left
# choice - which stimulus was chosen, e.g ('rosemary, pipes')
# S - the stimuli presented, e.g ('rosemary, pipes', 'lemon, WhBed')
# dimension1 - the first relevant dimension, e.g 'texture'

def make_ast_model_data(clean_data, mouse):

    # isolate data for each mouse
    tmp_data = clean_data.loc[(clean_data['mouse'] == mouse)].copy()

    # reset the index as useful for iterating later
    tmp_data.index = range(len(tmp_data))

    # block names
    tmp_block = tmp_data['block']

    # R, in alfredos code reward_loc = 1 = left
    # we need to flip this so it's on the correct side
    tmp_reward = tmp_data['reward_loc']
    pd.set_option('future.no_silent_downcasting', True) # to avoid warning
    tmp_reward.replace(to_replace=0, value=-1, inplace = True)
    tmp_reward_inv = tmp_reward*-1

    # replace all -1 with 0s (depending on how we want to treat ommisions)
    tmp_reward.replace(to_replace=-1, value=0, inplace = True)
    tmp_reward_inv.replace(to_replace=-1, value=0, inplace = True)

    # zip them up into 1 vector
    tmp_R = [[tmp_reward_inv, tmp_reward] for tmp_reward_inv, tmp_reward in zip(tmp_reward_inv, tmp_reward)]

    # Function to modify the array in df1 based on the value in df2
    #def modify_array(row1, row2):
    #    index = 1-row2  # This will be 0 or 1
    #    row1[index] = 0  # Modify the selected index in col1
    #    return row1

    # Apply the function to align both DataFrames
    #tmp_R = [
    #    modify_array(row1, row2) 
    #    for row1, row2 in zip(tmp_R, tmp_data['C'])
    #]

    ## S - need to cocatenate the each stimulus for the right and left stim, and make sure odour always first

    # first create empty lists for stim1 (right) and stim2 (left)
    tmp_s1 = []
    tmp_s2 = []

    # first loop over length of data
    for i in range(len(tmp_data['O1'])):
        
        # check if odours match, then in SD and no need to include in stim
        if tmp_data['O1_i'][i] == tmp_data['O2_i'][i]:
            # if chosen option = right
            if tmp_data['C'][i] == 0:
                # if texture of chosen option is 0 then right (s1) is O1
                if tmp_data['M'][i] == 0:
                    tmp_s1.append(tmp_data['T1_i'][i])
                    tmp_s2.append(tmp_data['T2_i'][i])
                else:
                    # M == 2 then right is M2 left is M1
                    tmp_s1.append(tmp_data['T2_i'][i])
                    tmp_s2.append(tmp_data['T1_i'][i])
            # if chosen option == 1 (left)
            else:
                # and the texture left is 0, then s2 (left) is m1
                if tmp_data['M'][i] == 0:
                    tmp_s1.append(tmp_data['T2_i'][i])
                    tmp_s2.append(tmp_data['T1_i'][i])
                else:
                    # otherwise M == 1 then s2 is m2
                    tmp_s1.append(tmp_data['T1_i'][i])
                    tmp_s2.append(tmp_data['T2_i'][i])

        # now lets do the same for when the textures match
        elif tmp_data['T1_i'][i] == tmp_data['T2_i'][i]:
        # if chosen option = right
            if tmp_data['C'][i] == 0:
                # if odour of chosen option is 0 then right (s1) is O1
                if tmp_data['O'][i] == 0:
                    tmp_s1.append(tmp_data['O1_i'][i])
                    tmp_s2.append(tmp_data['O2_i'][i])
                else:
                    # 0 == 2 then right is M2 left is M1
                    tmp_s1.append(tmp_data['O2_i'][i])
                    tmp_s2.append(tmp_data['O1_i'][i])
            # if chosen option == 1 (left)
            else:
                # and the odour left is 0, then s2 (left) is m1
                if tmp_data['O'][i] == 0:
                    tmp_s1.append(tmp_data['O2_i'][i])
                    tmp_s2.append(tmp_data['O1_i'][i])
                else:
                    # otherwise O == 1 then s2 is m2
                    tmp_s1.append(tmp_data['O1_i'][i])
                    tmp_s2.append(tmp_data['O2_i'][i])

        # now when neither match, meaning wer're out of simple discrimination 
        elif tmp_data['O1_i'][i] != tmp_data['O2_i'][i] and tmp_data['T1_i'][i] != tmp_data['T2_i'][i]:
            # if chose right
            if tmp_data['C'][i] == 0:

                # get the odour for the right (s1) left (s2) when chose right
                if tmp_data['O'][i] == 0:
                    tmp_s1_o = tmp_data['O1_i'][i]
                    tmp_s2_o = tmp_data['O2_i'][i]
                else:
                    tmp_s1_o = tmp_data['O2_i'][i]
                    tmp_s2_o = tmp_data['O1_i'][i]
                # get the texture for right and left when chose right
                if tmp_data['M'][i] == 0:
                    tmp_s1_m = tmp_data['T1_i'][i]
                    tmp_s2_m = tmp_data['T2_i'][i]
                else:
                    tmp_s1_m = tmp_data['T2_i'][i]
                    tmp_s2_m = tmp_data['T1_i'][i]
    
            # if chose left
            if tmp_data['C'][i] == 1:

                # get the odour for the right (s1) left (s2) when chose right
                if tmp_data['O'][i] == 0:
                    tmp_s2_o = tmp_data['O1_i'][i]
                    tmp_s1_o = tmp_data['O2_i'][i]
                else:
                    tmp_s2_o = tmp_data['O2_i'][i]
                    tmp_s1_o = tmp_data['O1_i'][i]
                # get the texture for right and left when chose right
                if tmp_data['M'][i] == 0:
                    tmp_s2_m = tmp_data['T1_i'][i]
                    tmp_s1_m = tmp_data['T2_i'][i]
                else:
                    tmp_s2_m = tmp_data['T2_i'][i]
                    tmp_s1_m = tmp_data['T1_i'][i]

            tmp_s1.append(tmp_s1_o + ',' + tmp_s1_m)
            tmp_s2.append(tmp_s2_o + ',' + tmp_s2_m)     


    tmp_S = [[tmp_s1, tmp_s2] for tmp_s1, tmp_s2 in zip(tmp_s1, tmp_s2)]
        
    # dimension 1
    tmp_d1 = tmp_data['relevant_dim'][0]
    if tmp_d1 == 'O':
        tmp_d1 = 'A'
    else:
        tmp_d1 = 'B'

    # Choice, Ani's model takes the chosen stimulus as the input for choice.
    choice = tmp_data['C'] # get choices and 1s (left) and 0s (right)
    tmp_choice = [] # empty list for storage

    # loop over tmp_S and take out the stimulus that was chosen, save in our list
    for i in range(len(choice)):
        tmp_choice.append(tmp_S[i][choice[i]])

    # get the group as well
    tmp_group = tmp_data['condition'][0]
        

    # combine into a dict
    model_data = {'R': tmp_R, 'choice': tmp_choice, 'S': tmp_S, 'dimension1': tmp_d1, 'condition': tmp_group}

    return model_data



####### function to sanity check the datalists for modelling
# picks a random subject from the datalist and checks if the data is correctly formatted for the model
def sanity_check(data):
    n_subs = len(data)
    random_subj = np.random.randint(n_subs)

    # let's get the data for the random subject
    tmp_data = data[random_subj]

    # how many trials do they have
    n_trials = len(tmp_data['R'])

    # let's get the last 8 trials and check if the rewarding stimulus is always the same (should be if they passed last stage)
    rewards = tmp_data['R'][-8:]
    choices = tmp_data['choice'][-8:]
    stims = tmp_data['S'][-8:]

    # check whether the last 8 trials were all rewarded
    x = 0 
    for i in range(len(stims)):
        index = stims[i].index(choices[i])
        reward = rewards[i][index]
        if reward == 0:
            x = 1

    # print the results 
    if x == 0:
        print('Passed: Last 8 choices are all rewarded!')
    else:
        print('Failed: Last 8 choices were not all rewarded, either the subject failed the last stage or something wrong in preprocessing')

    # check if dimensions_1 either A or B
    dim = tmp_data['dimension1'][0]
    if dim == 'A' or dim == 'B':
        print('Passed: Inital dimension coded correctly!')
    else:
        print('Failed: Initial dimension not coded correctly, check please.')