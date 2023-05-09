from predict import main_pred, init_state, ChargingStationPredictor, dict_tot
#from predict import *
from Fordonsdynamik import iterate, battery_temperature_change
from Konstanter import *
from cost import *
import math
import pandas as pd
from cost_regression import numpy_reg
from battery_time_regression import charging_powah

df = pd.read_csv(r'Algorithm\excel\chargers.csv')

def minimize_road_cost(road: int, TMs: dict, time_cost: float, profile: str) -> tuple:
    """ Räknar ut minimala kostnaden för en väg.
        returnerar kostnaden, en lista på chargers{id_char, time,....}"""
    current_point = 1
    total_cost_chargers = 0
    total_time = 0
    total_energy = 0
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
            total_energy = char_avail['energy_con'] + total_energy
            total_cost = total_cost_chargers + total_time * time_cost * profil[0] + total_energy/3600000 * profil[1]
            final_soc = char_avail['soc']
            break
        
        # Välj den bästa laddaren
        best_char, profil = choose_charger(char_avail, time_cost, profile)

        temp_diff, t_active = battery_temperature_change(best_char['index']-1, best_char['soc'], abs(best_char['temperature']-293))

        # Calculate the wanted values
        best_chargers[best_char['name']] = (best_char['soc charger'])
        total_time += best_char['charging time'] + best_char['drive time']
        total_driving_time += best_char['drive time']
        total_cost_chargers += best_char['charger cost']
        total_energy += best_char['energy consumption']

        current_point = best_char['index']
        soc = best_char['soc']
        print(f"Charging at: {best_char['name']} with SoC: {round(best_char['soc charger'],2)}% and preheating {round(t_active/60,2)} minutes before reaching charger")

    return total_cost, best_chargers, (total_time/3600, total_driving_time/3600), final_soc, total_energy #, timestops, timecharge?, mer?

def get_chargers_avail(idx_start: int, road: int, TMs: dict, soc: float) -> dict:
    """ Returns the availability of all chargers{capacity} in the selected span"""
    chargers, done = iterate(idx_start, road, soc)
    # Check if reach endpoint at Uppsala
    if done:
        print("Done with road:",road)
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
                char_avail[charger][cap] = \
                {
                    'soc_charger': value['soc_charger'],
                    'soc': value['soc'],
                    'availability': predictor.predict(steps=time_steps),
                    'state': state,
                    'time': chargers[charger]['time'],
                    'index': chargers[charger]['index'],
                    'distance': chargers[charger]['distance'],
                    'energy_consumption': chargers[charger]['energy_con'],
                    'temp_at_charger': chargers[charger]['temp_at_charger']
                }
            else:
                char_avail[charger] = {cap: \
                {
                    'soc_charger': value['soc_charger'],
                    'soc': value['soc'],
                    'availability': predictor.predict(steps=time_steps),
                    'state': state,
                    'time': chargers[charger]['time'],
                    'index': chargers[charger]['index'],
                    'distance': chargers[charger]['distance'],
                    'energy_consumption': chargers[charger]['energy_con'],
                    'temp_at_charger': chargers[charger]['temp_at_charger']
                }
                }
                                        
                                       

    return char_avail, False

def choose_charger(char_avail: dict, tc: float, profile: str) -> tuple[str, float, int]: 
    """ takes a dict of chargers, and calculates the cost of charging at each.
        returns a tuple with (id, cost)"""
    profiles = \
        {
            "time_minimized": (10,0.1,0.1),
            "energy_minimized": (0.1,10,0.1),
            "money_minimized": (0.1,0.1,10),
            "equally_minimized": (1,1,1)
        }
    profile = profiles[profile+"_minimized"]
    time_cost_mult = profile[0]
    energy_cost_mult = profile[1]
    money_cost_mult = profile[2]
    best_charger = 0
    best_charger_cost = 0
    a = numpy_reg()
    charging_power = charging_powah()
    for charger in char_avail:   # {charger_name: {50: (soc_50, state_predict[1xn]_50), 45: (soc_45, state_predict[1xn]_45)}}
        for cap, value in char_avail[charger].items():
            soc_charger = value['soc_charger']
            soc = value['soc']
            avail = value['availability']
            state = value['state']
            drive_time = value['time']
            index = value['index']
            distance = value['distance']
            energy_consumption = value['energy_consumption']
            temp_at_charger = value['temp_at_charger']

            # TODO maybe... lägg till förarprofiler som värderar de olika kostnaderna olika högt?
            ## Kolla kostnad         kr
            cost_el = Func_price_from_capa(cap, a)     # Löser sen /jakob_henrik
            tot_el, time_charge = Func_el_consum_and_time(soc_charger, cap, charging_power)      # Hampus gör idag 24/4
            tot_cost_el = cost_el * tot_el * money_cost_mult
        
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
            total_cost = (tot_cost_el * energy_cost_mult + tot_cost_time * time_cost_mult) * soc_cost / (soc_amount * avail_factor)

            # Checks if this is the best charger
            if total_cost < best_charger_cost or best_charger == 0:
                best_charger_cost = total_cost
                best_charger = \
                {
                    'name': charger,
                    'soc': soc,
                    'soc charger': soc_charger,
                    'charger cost': total_cost,
                    'drive time': drive_time,
                    'charging time': time_charge,
                    'index': index,
                    'distance': distance,
                    'energy consumption': energy_consumption,
                    'temperature': temp_at_charger 
                }


    return best_charger, profile
        

def main():
    """ Huvudfunc, kör igenom alla vägar, och returnerar bästa väg utifrån kostnad. 
    *Ger även alla laddstationer man stannar vid""" 
    TMs = main_pred()
    profile = input("Which cost do you want to minimize? (time, energy, money), leave empty for equally minimized: ")
    if profile == "":
        profile = "equally"
    roads = [1, 2, 3]
    total_road_time = [0,0,0]
    costs = [0,0,0]
    total_energy = [0,0,0]
    final_socs = [0,0,0]
    chargersList = [{}, {}, {}]
    time_cost = 10  #ger bara ett nummer för tester
    min_cost, chargersList[0], total_road_time[0], final_socs[0], total_energy[0] = minimize_road_cost(roads[0], TMs, time_cost, profile)       # returns the cost of choosing that road
    costs[0] = min_cost
    best_road_idx = 0

    for i in range(1, len(roads)):
        tot_cost, chargersList[i], total_road_time[i], final_socs[i], total_energy[i] = minimize_road_cost(roads[i], TMs, time_cost, profile)
        costs[i] = tot_cost
        if tot_cost < min_cost:
            min_cost = tot_cost
            best_road_idx = i
    print(f"Minimum cost: {min_cost}, \n Charger list:\n {chargersList[0]} \n {chargersList[1]} \n {chargersList[2]}, \n Total road time: {total_road_time}, \n Final socs: {final_socs}, \n Total energy: {total_energy}, \n Costs: {costs}")
    return roads[best_road_idx], min_cost


def testing_func():
    TMs = main_pred()


if __name__ == "__main__":
    print(main())
    #testing_func()