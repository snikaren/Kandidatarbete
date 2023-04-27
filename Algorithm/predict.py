import numpy as np
import pandas as pd
import sys
import math
import json
import random

def init_state(name: str, capacity: int, tot_dict: dict):
    """ Creates the initial state for a charging station"""
    # create a pandas series
    my_series = pd.Series(tot_dict[name][capacity])

    # get the count of occurrences of each unique value
    value_counts = my_series.value_counts(sort=True)
    
    distribution_df = pd.DataFrame({'Value': value_counts.index, 'Count': value_counts.values})
    distribution_df['procent'] = distribution_df['Count'] / sum(distribution_df['Count'])

    return distribution_df['Value'] ,distribution_df['procent']



def dict_tot():
    """ returns a dictonary for the charching stations containing the availability over the measured period"""
    f = open(r'Algorithm\Info.json')
    f2 = open(r'Algorithm/avail2023-03-13_16-43-55.json')
    info = json.load(f)
    avail = json.load(f2)
    tot_dict = {}
    # Går igenom alla chargers i info
    for charger in info:
        charger_name = charger['charger']
        tot_dict[charger_name] = {}
        capacitites = [i["capacity"] for i in charger["total_chargers"]]
        for idx, cap in enumerate(capacitites):  
            try:                   
                for k in range(len(avail[charger_name])):
                    if len(avail[charger_name][k][1]) == len(capacitites):
                        av = avail[charger_name][k][1][idx]
                        total_avail = int(av)
                        if cap in tot_dict[charger_name].keys():
                            tot_dict[charger_name][cap].append(total_avail)
                        else:
                            tot_dict[charger_name][cap] = [total_avail]
            except:
                print(f"Charger: {charger_name} not working")
        
    return tot_dict

    # https://stackoverflow.com/questions/47297585/building-a-transition-matrix-using-words-in-python-numpy
def tm_one_charger(charger_dict: dict):
    """ Returns the tranition matrix, for a given charging station """
    trans = pd.crosstab(pd.Series(charger_dict[1:],name='Next'),
                        pd.Series(charger_dict[:-1],name='Current'),normalize=1)
    return trans

def remove_recursive_points(t_dict: dict):
    """ Func that checks if a matrix has a recursive point, 
    and changes that column to have a 10% chance to reference another value"""
    for charger, caps in t_dict.items():
        for cap, t in caps.items():
            if len(t) > 1:
                for idx_row, row in t.iterrows():     # i for each row
                    for idx_col, col in enumerate(t.columns):
                        if row.iloc[idx_col] == 1:
                            r = random.choice([i for i in t.index if i != row.name])
                            t_dict[charger][cap].at[r, col] = 0.1
                            t_dict[charger][cap].at[idx_row, col] = 0.9
    return t_dict

def main_pred():
    dt = dict_tot()
    d = {}
    for charger in dt:
        d[charger] = {}
        for cap in dt[charger]:
            d[charger][cap] = tm_one_charger(dt[charger][cap])
    return remove_recursive_points(d)

""" A Class that runs the availability probabillity for a given station and time intervall"""
class ChargingStationPredictor:
    
    def __init__(self, states: list, transition_matrix, initial_state_distribution: list):
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_state_distribution = initial_state_distribution
        
    def predict(self, steps=1):
        current_state_distribution = self.initial_state_distribution
        
        for i in range(steps):
            next_state_distribution = np.dot(current_state_distribution, self.transition_matrix)
            current_state_distribution = next_state_distribution
            
        return current_state_distribution

# Gamal testfunc för att testa Class
def test_pred():
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

""" Func that runs the class CSP the correct amount of timesteps"""
def predict_avail(charger, timesteps, initial_state, trans_matrix):
    predictor = ChargingStationPredictor(charger, trans_matrix, initial_state)
    predictor.predict(steps=timesteps)
    
if __name__ == '__main__':
    print(dict_tot())
    #print("22mr3", dict_tot()['yrekr7'])
    