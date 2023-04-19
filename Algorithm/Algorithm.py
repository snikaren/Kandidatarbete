from predict import *
from Fordonsdynamik import *

# Räknar ut minimala kostnaden för en väg
def minimize_road_cost(road, time_cost):
    # Simuluera fram tills vi måste ladda 
    char_avail = get_chargers_avail(1, road)    # returns dict med charge_id(soc, avail)
    
    # Välj den bästa laddaren

        # Kör till punkten och ladda
        # ladda vid denna punkt
            # Kör hampus program
        # Få ut ny tid, plats och SOC
    # REPEAT MADDA FACKA



    # return: tid, soc, pengar

def get_chargers_avail(idx_start, road):
    chargers = iterate(idx_start,road)
    # returns: charge_dict[charger] = (soc, total_time)
    char_avail = {}
    for charger, value in chargers.items():
        current_avail = ca_func(charger)
        char_avail[charger] = (value[0], predict.func(charger, value[1], current_avail))     # Returns list with predicted availability
    return char_avail

def choose_charger(avail_dict, time_cost):
    for charger, value in avail_dict:   #value(SoC, Avail)
        soc = value[0]
        for avail in value[1]:
            ## Kolla kostnad         kr
            capacity = capa_dict[avail]
            cost_el = Func_price_from_capa (capacity)
            tot_el = Func_el_consum(soc, capacity)
            tot_cost_el = cost_el * tot_el 
        
            ## kolla tid att ladda   tid->kr
            time_charge = Func_time_charge
            tot_cost_time = time_cost * time_charge
        
            ## Kolla avail           true/false      
            average_avail, tot_avail = avail(nånting)      
            avail_antal = average_avail
            avail_procent = average_avail/tot_avail
            # Räkna ut en faktor som används för att väga procent mot antal
            faktor = 5          
            avail_factor = avail_procent*faktor + avail_antal
            total_cost = (tot_cost_el + tot_cost_time) / avail_factor

        


def main():
    min_cost =  minimize_road_cost(roads[0])
    best_road = roads[0]
    for road in roads[1:]:
        tot_cost = minimize_road_cost(road)
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road = road







if __name__ == "__main__":
    main()