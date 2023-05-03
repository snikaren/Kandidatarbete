from predict import main_pred, init_state, ChargingStationPredictor, dict_tot
#from predict import *
from Fordonsdynamik import iterate
from cost import *
import math
import pandas as pd
from cost_regression import numpy_reg

df = pd.read_csv(r'Algorithm\excel\chargers.csv')

def minimize_road_cost(road: int, chargers: dict, TMs: dict, time_cost: float) -> tuple:
    """ Räknar ut minimala kostnaden för en väg.
        returnerar kostnaden, en lista på chargers{id_char, time,....}"""
    current_point = 1
    total_cost = 0
    while True:
        
        # Simuluera fram tills vi måste ladda 
        char_avail = get_chargers_avail(current_point, road, TMs)    # returns dict med charge_id(soc, avail)
        if char_avail == 0:
            break
        
        # Välj den bästa laddaren       # RETURNERAR JUST NU EN LISTA MED KOSTNADEN???
        best_char, best_char_cost, index = choose_charger(char_avail, time_cost)
        #print(best_char, best_char_cost)

        # calculation on the choosen charger
        # chargers[best_char] = tiden_dit       ## fattar inte rikitigt vad som vill fås ut här???
            # Räkna ut när batterivärmning behöver startas
            # Kör till punkten och ladda
            # ladda vid denna punkt
                # Kör hampus program
            # total_cost =+ cost_trip
            # Få ut ny tid, plats och SOC - NÄR vi nått nästa punkt
        current_point = index
    # REPEAT med (plats, TMs, tc)

 
    return total_cost, (best_char, best_char_cost) #, timestops, timecharge?, mer?

def get_chargers_avail(idx_start: int, road: int, TMs: dict) -> dict:
    """ Returns the availability of all chargers{capacity} in the selected span"""
    chargers, done = iterate(idx_start, road)
    # returns: charge_dict[charger] = (soc, total_time)
    
    # Check if reach endpoint
    if done:
        #pass
        print("done")
        return 0
    
    char_avail = {}
    " Går igenom alla chargers och dess olika kapaciteter. "
    for charger, value in chargers.items():
        for cap in TMs[charger]:
            # set up for pred
            state, initial_state = init_state(charger, cap) 
            trans_matrix = TMs[charger][cap]
            time_steps = math.floor(value[1]/60/30)
            predictor = ChargingStationPredictor(state, trans_matrix, initial_state)

            # Runs the predictor the correct amount of steps
            # (soc, state, avail)
            if charger in char_avail:
                char_avail[charger][cap] = (value[0], predictor.predict(steps=time_steps), state, chargers[charger][2])
            else:
                char_avail[charger] = {cap: (value[0], predictor.predict(steps=time_steps), state, chargers[charger][2])}

    return char_avail

def choose_charger(char_avail: dict, tc: float) -> tuple: 
    """ takes a dict of chargers, and calculates the cost of charging at each.
        returns a tuple with (id, cost)"""
    best_charger = 0
    best_charger_cost = 0
    a = numpy_reg()
    for charger in char_avail:   # {charger_name: {50: (soc_50, state_predict[1xn]_50), 45: (soc_45, state_predict[1xn]_45)}}
        for cap, value in char_avail[charger].items():
            soc = value[0]
            avail = value[1]
            state = value[2]
            index = value[3]

            # TODO funktionerna price_from_capa, el_consum, time_charge
            # TODO maybe... lägg till förarprofiler som värderar de olika kostnaderna olika högt?
            ## Kolla kostnad         kr
            cost_el = Func_price_from_capa(cap, a)     # Löser sen /jakob_henrik
            tot_el = Func_el_consum(soc, cap)      # Hampus gör idag 24/4      # Kan flyttas till utanför for-loop
            tot_cost_el = cost_el * tot_el 
        
            ## kolla tid att ladda   tid->kr
            time_charge = Func_time_charge(soc, cap)           # Lös från FD
            tot_cost_time = tc * time_charge

            ## Kolla vad SOC är och vikta från det
            soc_cost = func_soc_cost(soc)
        
            ## Kolla avail           true/false      
            avail_procent, avail_num = get_avail_value(avail, state)       # (0-1), (numerical)
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 4
            avail_factor = avail_procent*faktor + avail_num
            total_cost = (tot_cost_el + tot_cost_time + soc_cost) / avail_factor
            if best_charger == 0:
                best_charger = charger
                best_charger_cost = total_cost
            elif total_cost < best_charger_cost:
                best_charger = charger
                best_charger_cost = total_cost

    return best_charger, best_charger_cost, index
        

def main():
    """ Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. 
    *Ger även alla laddstationer man stannar vid""" 
    TMs = main_pred()
    roads = [1, 2, 3]
    chargersList = [{}, {}, {}]
    time_cost = 10  #ger bara ett nummer för tester
    min_cost, chargersList[0] = minimize_road_cost(roads[0], chargersList[0], TMs, time_cost)       # returns the cost of choosing that road
    best_road_idx = 0
    return chargersList[0]
    for i in range(1, len(roads)):
        tot_cost, chargersList[i] = minimize_road_cost(roads[i], chargersList[i], TMs, time_cost)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road_idx = i


def testing_func():
    TMs = main_pred()


if __name__ == "__main__":
    print(main())
    #testing_func()