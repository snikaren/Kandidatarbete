## Skriver kostnadsfunktionerna här ##
# Bör de alla returnera en float eller något annat /Henrik
from cost_regression import predict
from battery_time_regression import main

def Func_price_from_capa(cap: int, a) -> float:
    """ Func that calculates the price of electricity given a charging capacity"""
    multiple = 1
    cost = predict(a, cap)
    return cost*multiple

def Func_el_consum_and_time(soc: float, cap: int, charging_powah) -> tuple[float, float]:
    """ Func that calculates the energy and time needed to charge a battery, given its capacity and the SOC of the car"""
    tot_charge, time_charge, battery_temp = main(soc, cap, charging_powah)
    print(soc, cap)
    return tot_charge, time_charge*3600

def func_soc_cost(soc: float) -> float:
    """ Func that adds a cost if we charge with high soc"""
    multiple_cost = 1       # just to show we can change the balancing
    return (soc - 10) * multiple_cost

def get_avail_value(avail: list, state: list) -> tuple:
    """ Func that given a avail_list and a state_list, calculates the availability in procent and total numerical chargers"""
    tot_char_avail = 0
    for i in range(len(avail)):
        tot_char_avail += avail[i]*state[i]
    proc_avail = tot_char_avail/state[len(state)-1]
    return proc_avail, tot_char_avail
    
    