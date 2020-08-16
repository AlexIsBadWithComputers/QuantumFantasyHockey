import pandas as pd
import numpy as np

'''
Simple function to create a binary "cost" function for each 
player in each position. This is not a true cost however, as this 
instead represents a position constraint equation of how many
players we are allowed to draft in each category. 
'''

def cost_filler(row):
    # Defence, Forwards, and Goalies 
    poskey = {'D':'D', 'L':'F', 'R':'F', 'C':'F', 'G':'G'}
    positions = ['D', 'F', 'G']
    pos = row.position
    tdict = {}
    tdict[poskey[pos]] = 1
    # Honestly, this was probably more work than just setting things
    # to zero with if statements, but lambda functions are sometimes
    # fun 
    zero = list(filter(lambda x: x != poskey[pos], positions))
    for p in zero:
        tdict[p] = 0
    
    return [tdict[p] for p in positions]


def linear_terms(returns, costs, num_position, sign = 1, lamb = 1):
    '''
    This function creates a python dictionary containing our linear terms, 
    i.e. the terms like sum_i c_i * x_i, where x_i is our binary
    player vector, and c_i is a multiplicative constant for this term.
    
    Inputs: returns -- > A vector of player scores 
            costs   --> The data frame as created by cost_filler
            num_position --> How many players occupy each position
            sign --> sign to change it from minimization/maximization 
            lamb --> lagrange multiplyer (not required)
            
    Returns: linear_dict --> A dictionary of our linear terms of the form
                             PlayerName:float 
    
    '''
    linear_dict = {}
    for name in returns.keys():
        name_cost = costs.loc[name]
        # Here we do our traditional Markowitz returns estimation
        linear_dict[name] = sign * returns[name]
        
        # Now we are adding up terms which represent our constraints
        val = 0
        for key in num_position:
            val += num_position[key] * name_cost[key] 
        # If I ignore the factor of two this works as expected,
        # double counting? 
        linear_dict[name] += sign * lamb * val
        
    
    return linear_dict

def quadratic_terms(covariance, costs, lamb=1):
    '''
    This is the function that generates the quadratic terms of our model. 
    In this case the traditional covariance used for Markowitz portfolio 
    optimization, as well as the other terms required for constraint satisfaction
    
    Inputs: covariance --> A NxN matrix of player covariances where N is the number
                           of players
            costs --> The data frame created by cost_filler
    
    Returns: dict_out --> A dictionary of our quadratic terms (edge weights) 
                          of the form (PlayerName1, PlayerName2): float
    '''
    
    
    dict_out = {}
    total = set(list(covariance))
    visited = set()
    for i in list(covariance):
        # Check to see if we've visited our node alread
        # if we have no point in looping through it again
        for j in list(total ^ visited): 
            dict_out[(i, j)] = covariance[i][j] 
            
            for c in list(costs):
                dict_out[(i, j)] += lamb * costs[c][i] * costs[c][j]
    
        visited.add(i)
        
    return dict_out