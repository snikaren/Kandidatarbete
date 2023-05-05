from predict import main_pred, init_state, ChargingStationPredictor, dict_tot
#from predict import *
from Fordonsdynamik import iterate
from Konstanter import *
from cost import *
import math
import pandas as pd
from cost_regression import numpy_reg
from battery_time_regression import charging_powah

df = pd.read_csv(r'Algorithm\excel\chargers.csv')

def minimize_road_cost(road: int, TMs: dict, time_cost: float) -> tuple:
    """ Räknar ut minimala kostnaden för en väg.
        returnerar kostnaden, en lista på chargers{id_char, time,....}"""
    current_point = 1
    total_cost_chargers = 0
    total_time = 0
    soc = 100 * max_battery/Battery_joules
    total_driving_time = 0
    best_chargers = {}
    while True:
        
        # Simuluera fram tills vi måste ladda, hämtar laddare i området, och ger parametrarna för dessa
        char_avail, done = get_chargers_avail(current_point, road, TMs, soc)

        # Check if we reached end point
        if done:
            total_time += char_avail['time']
            total_driving_time += char_avail['time']
            total_cost = total_cost_chargers + total_time * time_cost
            final_soc = char_avail['soc']
            break
        
        # Välj den bästa laddaren
        best_char, best_char_cost, index, time_charge, time_drive, soc, soc_charger, highway_dist = choose_charger(char_avail, time_cost)

        # calculate the wanted values
        best_chargers[best_char] = (soc_charger, highway_dist)
        total_time += time_charge + time_drive
        total_driving_time += time_drive
        total_cost_chargers += best_char_cost

        current_point = index

    return total_cost, best_chargers, (total_time/3600, total_driving_time/3600), final_soc #, timestops, timecharge?, mer?

def get_chargers_avail(idx_start: int, road: int, TMs: dict, soc: float) -> dict:
    """ Returns the availability of all chargers{capacity} in the selected span"""
    chargers, done = iterate(idx_start, road, soc)
    # Check if reach endpoint at Uppsala
    if done:
        print("Done with road:",road,", params at end:",chargers)
        return chargers, True
    
    char_avail = {} 
    " Går igenom alla chargers och dess olika kapaciteter. "
    for charger, value in chargers.items():
        for cap in TMs[charger]:
            # set up for pred
            state, initial_state = init_state(charger, cap) 
            trans_matrix = TMs[charger][cap]
            time_steps = math.floor(value['time']/60/30)
            predictor = ChargingStationPredictor(state, trans_matrix, initial_state)

            # Runs the predictor the correct amount of steps
            # (soc, state, avail)
            if charger in char_avail:
                char_avail[charger][cap] = (value['soc_charger'], value['soc'], predictor.predict(steps=time_steps), state, chargers[charger]['time'], chargers[charger]['index'], chargers[charger]['highway_dist'])
            else:
                char_avail[charger] = {cap: (value['soc_charger'], value['soc'], predictor.predict(steps=time_steps), state, chargers[charger]['time'], chargers[charger]['index'], chargers[charger]['highway_dist'])}

    return char_avail, False

def choose_charger(char_avail: dict, tc: float) -> tuple[str, float, int]: 
    """ takes a dict of chargers, and calculates the cost of charging at each.
        returns a tuple with (id, cost)"""
    best_time_charge = 0
    best_time_drive = 0
    best_charger = 0
    best_charger_cost = 0
    a = numpy_reg()
    charging_power = charging_powah()
    for charger in char_avail:   # {charger_name: {50: (soc_50, state_predict[1xn]_50), 45: (soc_45, state_predict[1xn]_45)}}
        for cap, value in char_avail[charger].items():
            soc_charger = value[0]
            soc = value[1]
            avail = value[2]
            state = value[3]
            drive_time = value[4]
            index = value[5]
            charger_dist = value[6]

            # TODO maybe... lägg till förarprofiler som värderar de olika kostnaderna olika högt?
            ## Kolla kostnad         kr
            cost_el = Func_price_from_capa(cap, a)     # Löser sen /jakob_henrik
            tot_el, time_charge = Func_el_consum_and_time(soc_charger, cap, charging_power)      # Hampus gör idag 24/4
            tot_cost_el = cost_el * tot_el 
        
            ## kolla tid att ladda   tid->kr
            # time_charge = Func_time_charge(soc, cap)           # Lös från FD
            tot_cost_time = tc * time_charge

            ## Kolla vad SOC är och vikta från det
            soc_cost = func_soc_cost(soc_charger)
            soc_amount = 80 - soc_charger
        
            ## Kolla avail           true/false      
            avail_procent, avail_num = get_avail_value(avail, state)       # (0-1), (numerical)
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 3
            avail_factor = avail_procent*faktor + avail_num
            total_cost = (tot_cost_el + tot_cost_time) * soc_cost / (soc_amount * avail_factor)

            # Checks if this is the best charger
            if total_cost < best_charger_cost or best_charger == 0:
                best_charger = charger
                best_soc = soc
                best_soc_charger = soc_charger
                best_charger_cost = total_cost
                best_time_drive = drive_time
                best_time_charge = time_charge
                best_index = index
                best_charger_dist = charger_dist


    return best_charger, best_charger_cost, best_index, best_time_charge, best_time_drive, best_soc, best_soc_charger, best_charger_dist
        

def main():
    """ Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. 
    *Ger även alla laddstationer man stannar vid""" 
    TMs = main_pred()
    roads = [1, 2, 3]
    total_road_time = [0, 0, 0]
    final_socs = [0,0,0]
    chargersList = [{}, {}, {}]
    time_cost = 10  #ger bara ett nummer för tester
    min_cost, chargersList[0], total_road_time[0], final_socs[0] = minimize_road_cost(roads[0], TMs, time_cost)       # returns the cost of choosing that road
    best_road_idx = 0

    for i in range(1, len(roads)):
        tot_cost, chargersList[i], total_road_time[i], final_socs[i] = minimize_road_cost(roads[i], TMs, time_cost)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road_idx = i
    print(f"Minimum cost: {min_cost}, \n Charger list:\n {chargersList[0]} \n {chargersList[1]} \n {chargersList[2]}, \n Total road time: {total_road_time}, \n Final socs: {final_socs}")
    return roads[best_road_idx], min_cost


def testing_func():
    TMs = main_pred()


if __name__ == "__main__":
    print(main())
    #testing_func()