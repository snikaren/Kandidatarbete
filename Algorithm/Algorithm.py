from predict import main_pred, init_state, predict_avail, ChargingStationPredictor, dict_tot
#from predict import *
from Fordonsdynamik import iterate
from cost import *
import math
import pandas as pd

df = pd.read_csv(r'Algorithm\excel/chargers.csv')

tot_dict = dict_tot()
" global variabel dict_tot()"

def minimize_road_cost(road, chargers, TMs, time_cost):
    """ Räknar ut minimala kostnaden för en väg"""
    # Simuluera fram tills vi måste ladda 
    current_point = 1
    total_cost = 0
    while True:
        
        char_avail = get_chargers_avail(current_point, road, TMs)    # returns dict med charge_id(soc, avail)
        if char_avail == 0:
            break
        
        # Välj den bästa laddaren       # RETURNERAR JUST NU EN LISTA MED KOSTNADEN???
        best_char = choose_charger(char_avail, TMs, time_cost)

        # calculation on the choosen charger
        # chargers[best_char] = tiden_dit       ## fattar inte rikitigt vad som vill fås ut här???
            # Räkna ut när batterivärmning behöver startas
            # Kör till punkten och ladda
            # ladda vid denna punkt
                # Kör hampus program
            # total_cost =+ cost_trip
            # Få ut ny tid, plats och SOC - NÄR vi nått nästa punkt
        
    # REPEAT med (plats, TMs, tc)

 
    return total_cost, chargers #, timestops, timecharge?, mer?

def get_chargers_avail(idx_start, road, TMs):
    """ Returns the availability of all chargers{capacity} in the selected span"""
    chargers, done = iterate(idx_start, road)
    # returns: charge_dict[charger] = (soc, total_time)
    
    # Check if reach endpoint
    if done:
        pass
        #return 0
    
    print(chargers)

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

    
    print("avail end")
    return char_avail


def choose_charger(avail_dict, tc):
    """ takes a dict of chargers, and calculates the cost of charging at each.
        returns the (best/list of value) ### VILKEN VILL VI HA????"""
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
            time_charge = Func_time_charge(soc, cap)           # Lös från FD
            tot_cost_time = tc * time_charge
        
            ## Kolla avail           true/false      
            average_avail, tot_avail = get_avail_value(avail)
            avail_antal = average_avail
            avail_procent = average_avail/tot_avail
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 5          
            avail_factor = avail_procent*faktor + avail_antal
            total_cost = (tot_cost_el + tot_cost_time) / avail_factor

            total_cost_list.append(total_cost)

    # Ska vi returnera listan, eller bara bästa värdet?
    # Vi är ju egenltigen bara intresserade av värdet, samt vilket laddare
    return total_cost_list
        

def main():
    """ Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. 
    *Ger även alla laddstationer man stannar vid""" 
    # skapa en funk som sparar cvs som roads[]
    TMs = main_pred()
    roads = [1, 2, 3]
    chargersList = [{}, {}, {}]        # Gissar att denna ska ligga utanför i main?? // Henrik
    time_cost = 10  #ger bara ett nummer för tester
    min_cost, chargersList[0] = minimize_road_cost(roads[0], chargersList[0], TMs, time_cost)       # returns the cost of choosing that road
    best_road_idx = 0
    for i in range(1, len(roads)):
        tot_cost, chargersList[i] = minimize_road_cost(roads[i], chargersList[i], TMs, time_cost)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road_idx = i


def testing_func():
    TMs = main_pred()




if __name__ == "__main__":
    main()
    #testing_func()