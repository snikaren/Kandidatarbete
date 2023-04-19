import numpy as np
import pandas as pd
import sys
import math
import json
import random

# Code for predicting the availability of charging stations
# Ideer: 
#   Bayesian predition (Naive?)
#   Markov chain
#   Classification through prediction 

""" Går igenom avail_datan för varje laddare, och returnerar listor med 
    Total chargers
    Average availability
    procentual availabillity"""
""" Osäker på om denna behövs, men sparar sålänge """

def calc_trans_matrix():
    f = open('info.json')
    f2 = open(r'C:\Users\Henrik\Documents\Chalmers\PythonProg\Charging_Station_availability\avail2023-03-13_16-43-55.json')
    info = json.load(f)
    avail = json.load(f2)
    ave_list = []
    pro_list = []
    # Går igenom alla chargers i info
    for i, charger in enumerate(info):
        total_chargers = 0
        tot = charger["total_chargers"]
        for j in tot:
            total_chargers += int(j['count'])
        total_avail = 0
        charger_name = charger['charger']
        # samlar tillgängligheten för hela laddstationen
        # Kollar att laddstationen finns, om ej hoppar över
        try:
            for k in range(len(avail[charger_name])):
                for n in avail[charger_name][k][1]:
                    total_avail += int(n)
            average_avail = total_avail/len(avail[charger_name])
            procent_avail = average_avail/total_chargers
            ## Gör listor eller biblotek som sparar denna datan
        except:
            print(f"Charger: {charger_name} not in log")
    return average_avail, procent_avail, total_chargers


""" returns a dictonary for the charching stations containing the availability over the measured period"""
def dict_tot():
    f = open('info.json')
    f2 = open(r'C:\Users\Henrik\Documents\Chalmers\PythonProg\Charging_Station_availability\avail2023-03-13_16-43-55.json')
    info = json.load(f)
    avail = json.load(f2)
    tot_dict = {}
    # Går igenom alla chargers i info
    try:
        for charger in info:
            charger_name = charger['charger']       # Charger är 6 siffriga koden
            # Kollar att charge_id finns in avail
            # Om det finns, lägger till värdet i dict
            tot_dict[charger_name] = {}
            capacitites = [i["capacity"] for i in charger["total_chargers"]]
            for idx, cap in enumerate(capacitites):                     
                for k in range(len(avail[charger_name])):
                                # Resets the availabilty
                    if len(avail[charger_name][k][1]) == len(capacitites):
                        av = avail[charger_name][k][1][idx]
                        total_avail = int(av)
                        if cap in tot_dict[charger_name].keys():
                            tot_dict[charger_name][cap].append(total_avail)
                        else:
                            tot_dict[charger_name][cap] = [total_avail]
    except:
        print(f"Charger: {charger_name} not in log")
        
    return tot_dict

"""" Returns the tranition matrix, for a given charging station """
def tm_one_charger(charger_dict):
    # https://stackoverflow.com/questions/47297585/building-a-transition-matrix-using-words-in-python-numpy
    trans = pd.crosstab(pd.Series(charger_dict[1:],name='Next'),
                        pd.Series(charger_dict[:-1],name='Current'),normalize=1)
    return trans

def remove_recursive_points(t_dict):
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

class ChargingStationPredictor:
    
    def __init__(self, states, transition_matrix, initial_state_distribution):
        self.states = states
        self.transition_matrix = transition_matrix
        self.initial_state_distribution = initial_state_distribution
        
    def predict(self, steps=1):
        current_state_distribution = self.initial_state_distribution
        
        for i in range(steps):
            next_state_distribution = np.dot(current_state_distribution, self.transition_matrix)
            current_state_distribution = next_state_distribution
            
        return current_state_distribution
    
    
if __name__ == '__main__':
    main_pred()