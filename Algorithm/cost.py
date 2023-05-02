## Skriver kostnadsfunktionerna här ##
# Bör de alla returnera en float eller något annat /Henrik

def Func_price_from_capa(cap) -> float:
    """ Func that calculates the price of electricity given a charging capacity"""
    return 5

def Func_el_consum(soc, cap) -> float:
    """ Func that calculates the energy needed to charge a battery given its capacity and the SOC of the car"""
    return 5

def Func_time_charge(soc, cap) -> float:
    """ Func that calculates how long it takes to charge given a capacity and the SOC of the car"""
    return 5

def get_avail_value(avail: list, state: list) -> tuple:
    """ Func that given a avail_list and a state_list, calculates the availability in procent and total numerical chargers"""
    tot_char_avail = 0
    for i in range(len(avail)):
        tot_char_avail += avail[i]*state[i]
    #print(state[0])
    #print(avail)
    proc_avail = tot_char_avail/state[0]
    return proc_avail, tot_char_avail
    
    