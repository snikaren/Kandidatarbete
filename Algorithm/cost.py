## Skriver kostnadsfunktionerna här ##
# Bör de alla returnera en float eller något annat /Henrik
from cost_regression import predict
from battery_time_regression import main
import warnings

def Func_price_from_capa(cap: int, a) -> float:
    """ Func that calculates the price of electricity given a charging capacity"""
    multiple = 1
    cost = predict(a, cap)
    return cost*multiple

def Func_el_consum_and_time(soc: float, cap: int, charging_powah, battery_temp) -> tuple[float, float]:
    """ Func that calculates the energy and time needed to charge a battery, given its capacity and the SOC of the car"""
    tot_charge, time_charge, battery_temp = main(soc, cap, charging_powah, battery_temp)
    return tot_charge[0], time_charge*3600, battery_temp

def func_soc_cost(soc: float) -> float:
    """ Func that adds a cost if we charge with high soc"""
    if soc > 26:
        multiple_cost = (soc - 20)/ 3      # makes charging early bad
    else:
        multiple_cost = 2
    return soc * multiple_cost

def get_avail_value(avail: list, state: list) -> tuple[float, float]:
    """ Func that given a avail_list and a state_list, calculates the availability in procent and total numerical chargers"""
    tot_char_avail = 0
    state = state.tolist()
    for i in range(len(avail)):
        
        tot_char_avail += avail[i]*state[i]

    # A try/except filter to remove warnings that occour from tot_avail or state[x] being NaN or = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proc_avail = tot_char_avail/state[len(state)-1]
    except Exception as e:
        print(f"An exception occurred: {e}")
    """
    try:
        proc_avail = tot_char_avail/state[len(state)-1]
    except RuntimeWarning:
        print("runtimeWarning")
        """
    return proc_avail, tot_char_avail
    
    