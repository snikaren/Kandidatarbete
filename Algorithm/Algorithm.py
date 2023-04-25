from predict import main_pred, init_state, predict_avail, ChargingStationPredictor, dict_tot
#from predict import *
from Fordonsdynamik import iterate
from cost import *
import math
import pandas as pd

df = pd.read_csv('chargers.csv')
tot_dict = dict_tot()

""" Räknar ut minimala kostnaden för en väg"""
def minimize_road_cost(road, TMs, time_cost):
    # Simuluera fram tills vi måste ladda 
    current_point = 1
    total_cost = 0
    chargers = {1: {}, 2: {}, 3: {}}        # Gissar att denna ska ligga utanför i main?? // Henrik
    while True:
        
        char_avail = get_chargers_avail(current_point, road, TMs)    # returns dict med charge_id(soc, avail)
        if char_avail == 0:
            break
        
        # Välj den bästa laddaren
        best_char = choose_charger(char_avail, TMs, time_cost)
        chargers[road][best_char] = tiden_dit
            # Räkna ut när batterivärmning behöver startas
            # Kör till punkten och ladda
            # ladda vid denna punkt
                # Kör hampus program
            # total_cost =+ cost_trip
            # Få ut ny tid, plats och SOC - NÄR vi nått nästa punkt
        
    # REPEAT med (plats, TMs, tc)

 
    return total_cost, chargers #, timestops, timecharge?, mer?

""" Returns the availability of all chargers{capacity} in the selected span"""
def get_chargers_avail(idx_start, road, TMs):
    chargers, done = iterate(idx_start, road)
    # returns: charge_dict[charger] = (soc, total_time)
    
    # Check if reach endpoint
    if done:
        return 0
    
    char_avail = {}
    " Går igenom alla chargers och dess olika kapaciteter. "
    for charger, value in chargers.items():
        for cap in TMs[charger]:
            # Set up
            initial_state = init_state(charger, cap, tot_dict) 
            trans_matrix = TMs[charger][cap]
            time_steps = math.floor(value[1]/60/30)
            predictor = ChargingStationPredictor(charger, trans_matrix, initial_state)

            # Runs the predictor the correct amount of steps
            # (soc, avail)
            char_avail[charger][cap] = (charger[0], predictor.predict(steps=time_steps))

    return char_avail



def choose_charger(avail_dict, tc):
    total_cost_list = []
    for charger in avail_dict:   # {charger_name: {50: (soc_50, state_predict[1xn]_50), 45: (soc_45, state_predict[1xn]_45)}}
        for cap, value in charger:
            soc = value[0]
            avail = value[1]

            ## Kolla kostnad         kr
            cost_el = Func_price_from_capa(cap)     # Löser sen /jakob_henrik
            tot_el = Func_el_consum(soc, cap)      # Hampus gör idag 24/4      # Kan flyttas till utanför for-loop
            tot_cost_el = cost_el * tot_el 
        
            ## kolla tid att ladda   tid->kr
            time_charge = Func_time_charge              # Lös från FD
            tot_cost_time = tc * time_charge
        
            ## Kolla avail           true/false      
            average_avail, tot_avail = avail(nånting)
            avail_antal = average_avail
            avail_procent = average_avail/tot_avail
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 5          
            avail_factor = avail_procent*faktor + avail_antal
            total_cost = (tot_cost_el + tot_cost_time) / avail_factor

            total_cost_list.append(total_cost)

    return total_cost_list
        

""" Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. *Ger även alla laddstationer man stannar vid"""
def main():
    # skapa en funk som sparar cvs som roads[]
    TMs = main_pred()
    roads = [1, 2, 3]
    min_cost =  minimize_road_cost(roads[0], TMs)       # returns the cost of choosing that road
    best_road = roads[0]
    for road in roads[1:]:
        tot_cost = minimize_road_cost(road)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road = road


def testing_func():
    TMs = main_pred()




if __name__ == "__main__":
    #main()
    testing_func()