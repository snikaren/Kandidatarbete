from predict import main_pred, init_state, predict_avail, ChargingStationPredictor
#from predict import *
from Fordonsdynamik import *
import math
import pandas as pd

df = pd.read_csv('chargers.csv')
# Räknar ut minimala kostnaden för en väg
def minimize_road_cost(road, TMs, time_cost):
    # Simuluera fram tills vi måste ladda 
    char_avail = get_chargers_avail(1, road, TMs)    # returns dict med charge_id(soc, avail)
    
    # Välj den bästa laddaren
    best_char = choose_charger(char_avail, TMs, time_cost)
        # Kör till punkten och ladda
        # ladda vid denna punkt
            # Kör hampus program
        # Få ut ny tid, plats och SOC
    # REPEAT MADDA FACKA


    # return: tid, soc, pengar

""" Returns the availability of all chargers in the selected span"""
def get_chargers_avail(idx_start, road, TMs):
    chargers = iterate(idx_start,road)
    # returns: charge_dict[charger] = (soc, total_time)
    char_avail = {}
    " Går igenom alla chargers och dess olika kapaciteter. "
    for charger, value in chargers.items():
        # caps = tuple(map(int, df[df['name'] == charger]['capacity'].split(",")))          Behövs inte längre men sparar för säkerhet
        # Uses TMs to find the different capacities
        for cap in TMs[charger]:
            # Set up
            initial_state = init_state(charger, cap)        # start vektorn för charger
            trans_matrix = TMs[charger][cap]
            time_steps = math.floor(value[1]/60/30)
            predictor = ChargingStationPredictor(charger, trans_matrix, initial_state)
            char_avail[charger][cap] = (charger[0], predictor.predict(steps=time_steps))
        # char_avail[charger] = (value[0], predict_avail.func(charger, math.floor(value[1]/60/30), current_avail))     # Returns list with predicted availability
    return char_avail   



def choose_charger(avail_dict, tc):
    for charger, value in avail_dict:   #value(SoC, Avail)
        soc = value[0]
        for avail in value[1]:
            ## Kolla kostnad         kr
            capacity = capa_dict[charger][cap]
            cost_el = Func_price_from_capa (capacity)
            tot_el = Func_el_consum(soc, capacity)
            tot_cost_el = cost_el * tot_el 
        
            ## kolla tid att ladda   tid->kr
            time_charge = Func_time_charge
            tot_cost_time = tc * time_charge
        
            ## Kolla avail           true/false      
            average_avail, tot_avail = avail(nånting)      
            avail_antal = average_avail
            avail_procent = average_avail/tot_avail
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 5          
            avail_factor = avail_procent*faktor + avail_antal
            total_cost = (tot_cost_el + tot_cost_time) / avail_factor

        

""" Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. *Ger även alla laddstationer man stannar vid"""
def main():
    # skapa en funk som sparar cvs som roads[]
    TMs = main_pred()
    min_cost =  minimize_road_cost(roads[0], TMs)
    best_road = roads[0]
    for road in roads[1:]:
        tot_cost = minimize_road_cost(road)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road = road







if __name__ == "__main__":
    main()