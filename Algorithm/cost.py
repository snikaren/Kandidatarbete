## Skriver kostnadsfunktionerna här ##
# Bör de alla returnera en float eller något annat /Henrik
from cost_regression import predict

def Func_price_from_capa(cap: int, a) -> float:
    """ Func that calculates the price of electricity given a charging capacity"""
    multiple = 1
    cost = predict(a, cap)
    #print("kost", cost, " cap", cap)
    return cost*multiple

def Func_el_consum(soc: float, cap: int) -> float:
    """ Func that calculates the energy needed to charge a battery given its capacity and the SOC of the car"""
    return 5

def Func_time_charge(soc: float, cap: int) -> float:
    """ Func that calculates how long it takes to charge given a capacity and the SOC of the car"""
    return 5

def func_soc_cost(soc: float) -> float:
    """ Func that adds a cost if we charge with high soc"""
    multiple_cost = 1       # 
    return (soc - 10) * multiple_cost

def get_avail_value(avail: list, state: list) -> tuple:
    """ Func that given a avail_list and a state_list, calculates the availability in procent and total numerical chargers"""
    tot_char_avail = 0
    for i in range(len(avail)):
        tot_char_avail += avail[i]*state[i]
    proc_avail = tot_char_avail/state[len(state)-1]
    return proc_avail, tot_char_avail
    
    