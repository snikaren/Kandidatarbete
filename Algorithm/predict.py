import numpy as np
import pandas as pd
import sys
import math
import json
import random

def dict_tot() -> dict:
    """ Returns a dictonary for the charching stations containing the availability over the measured period"""

    tot_dict = {}  # Initialize an empty dictionary

    # Open and load the 'Info.json' file
    with open(r'Algorithm\Info.json') as f:
        info = json.load(f)

    # Open and load the 'avail2023-03-13_16-43-55.json' file
    with open(r'Algorithm/avail2023-03-13_16-43-55.json') as f2:
        avail = json.load(f2)

    # Iterate over each charger in the 'info' JSON data
    for charger in info:
        charger_name = charger['charger']  # Get the name of the charger
        tot_dict[charger_name] = {}  # Create an empty dictionary for the charger

        # Extract the capacities from 'total_chargers' for the current charger
        capacitites = [i["capacity"] for i in charger["total_chargers"]]

        # Iterate over each capacity and its corresponding index
        for idx, cap in enumerate(capacitites):
            try:
                # Iterate over each entry in 'avail' for the current charger
                for k in range(len(avail[charger_name])):
                    # Check if the length of the availability matches the number of capacities
                    if len(avail[charger_name][k][1]) == len(capacitites):
                        av = avail[charger_name][k][1][idx]  # Get the availability value
                        total_avail = int(av)  # Convert availability to an integer

                        # Check if the capacity exists in the charger's dictionary
                        if cap in tot_dict[charger_name].keys():
                            tot_dict[charger_name][cap].append(total_avail)
                        else:
                            tot_dict[charger_name][cap] = [total_avail]
            except:
                pass  # Ignore any exceptions that occur during the processing
    

    return tot_dict

def init_state(name: str, capacity: int, tot_dict: dict) -> tuple:
    """ Creates the initial state for a charging station"""
    
    #global tot_dict
    # create a pandas series
    my_series = pd.Series(tot_dict[name][capacity], dtype='float64')

    # get the count of occurrences of each unique value
    value_counts = my_series.value_counts(sort=True)
    
    distribution_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})
    distribution_df['procent'] = distribution_df['Count'] / sum(distribution_df['Count'])
    df = distribution_df.sort_values('Value').reset_index(drop = True)

    return df['Value'], df['procent']

def tm_one_charger(charger_dict: dict): # -> DataFrame
    """ Returns the tranition matrix, for a given charging station """
    trans = pd.crosstab(pd.Series(charger_dict[1:],name='Next',dtype='float64'),
                        pd.Series(charger_dict[:-1],name='Current',dtype='float64'),normalize=1)

    return trans

def remove_recursive_points(t_dict: dict) -> dict:
    """ Func that checks if a matrix has a recursive point, 
    and changes that column to have a 10% chance to reference another random value"""

    for charger, caps in t_dict.items():
        for cap, t in caps.items():
            if len(t) > 1:
                rows_to_update = t.index[t.eq(1).any(axis=1)]  # Get rows with at least one value of 1
                if len(rows_to_update) > 0:
                    r = random.choice(rows_to_update)
                    t.loc[r] = 0.1  # Set the selected row to 0.1 for all columns
                    t.loc[t.index != r] = 0.9  # Set other rows to 0.9 for all columns
    return t_dict

def main_pred(total_dict) -> dict:
    """ a func that returns the dict of all availability of all chargers in info, 
    with recuring points removed"""
    dt = total_dict
    d = {}
    for charger in dt:
        d[charger] = {}
        for cap in dt[charger]:
            d[charger][cap] = tm_one_charger(dt[charger][cap])
    return remove_recursive_points(d)

class ChargingStationPredictor:
    """ A Class that runs the availability probabillity for a given station and time intervall"""
    
    def __init__(self, states: list, transition_matrix, initial_state_distribution: list):
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_state_distribution = initial_state_distribution
        
    def predict(self, steps=1) -> list:
        """ a functiion that runs the state_dist 'steps' number of times to calculate the future distribution"""
        current_state_distribution = self.initial_state_distribution
        
        for i in range(steps):
            next_state_distribution = np.dot(current_state_distribution, self.transition_matrix)
            current_state_distribution = next_state_distribution
            
        return current_state_distribution

def test_pred():
    " Test function for showing how the class ChargingStationPredictor works. states are not needed"
    states = [0, 1, 2, 3, 4]
    transition_matrix = np.array([[0.3, 0.4, 0.2, 0.1, 0], 
                                [0.1, 0.3, 0.4, 0.2, 0], 
                                [0, 0.1, 0.3, 0.4, 0.2], 
                                [0.2, 0, 0.1, 0.3, 0.4], 
                                [0.4, 0.2, 0, 0.1, 0.3]])
    initial_state_distribution = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    

    predictor = ChargingStationPredictor(states, transition_matrix, initial_state_distribution)

    print(predictor.predict(steps=1))
    print(predictor.predict(steps=2))
    
#if __name__ == '__main__':
    #test_pred()