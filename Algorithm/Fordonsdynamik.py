from Konstanter import *
from charger_iteration import iterate_charger
import numpy as np
import pandas as pd
import time
import math as m
#import matplotlib.pyplot as plt
#from charger_iteration import *         #iterate_charger

# Dataframe uppsättning
df = ""
grads, speed, dists = 0,0,0
df_charge = pd.read_csv(r'Algorithm\excel\chargers.csv')

def init_df(route):
    """ grabs the correct cvs-file and make it the DataFrame of the iteration"""
    global df, grads, speed, dists
    if route == 1:
        df = pd.read_csv(r'Algorithm\excel\rutt_1.csv')
    elif route == 2:
        df = pd.read_csv(r'Algorithm\excel\rutt_2.csv')
    else:
        df = pd.read_csv(r'Algorithm\excel\rutt_3.csv')

    grads = df['gradient to next']
    speed = df['speed limit']
    dists = df['dist to next[Km]']

#  Läser in csv-filen
data = np.loadtxt(r'Algorithm\excel\AccelerationsTider.csv', delimiter=";", skiprows=1)
acc_dict = dict()

#  Dict med hastighet som key och acc_tid som value
for i in range(len(data)):
    acc_dict[data[i, 0]] = data[i, 1]


# Resulting acceleration of car
def acc_tot(idx):
    velo_current = float(Current_pd(speed, idx))
    velo_previous = float(Current_pd(speed, indx(prev_point(idx))))

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return abs(velo_current - velo_previous)/(3.6*(acc_dict[velo_current] - acc_dict[velo_previous]))


def force_tot(idx):
    velo_current = float(Current_pd(speed, idx))
    velo_previous = float(Current_pd(speed, indx(prev_point(idx))))

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return m_bil*(velo_current - velo_previous) / (3.6 * (acc_dict[velo_current] - acc_dict[velo_previous]))


# Dragforce
def force_air(idx, mode):
    velo_current = float(Current_pd(speed, idx)) / 3.6
    velo_previous = float(Current_pd(speed, indx(prev_point(idx)))) / 3.6
    if mode == 0:
        #  Mean of final velocity and starting velocity
        return den_air * cd * area_front * ((velo_current + velo_previous) / 2) ** 2 / 2
    elif mode == 1:
        #  Final velocity
        return den_air * cd * area_front * velo_current ** 2 / 2


def force_elevation(idx):
    elev_current = float(Current_pd(grads, idx))
    return m_bil * g * (np.sin(elev_current) + c_r * np.cos(elev_current))


def force_traction_acc(idx):
    return force_tot(idx) + force_air(idx, 0) + force_elevation(idx)


def force_traction_const(idx):
    return force_air(idx, 1) + force_elevation(idx)


## Beräkningar av effekt och arbete från Acc ##
# Beräknar tiden det tar att accelerera mellan två olika hastigheter
def time_acc(idx):
    #  /3.6 omvandlar till m/s
    velo_current = float(Current_pd(speed, idx))/3.6
    velo_previous = float(Current_pd(speed, indx(prev_point(idx))))/3.6

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        return abs(velo_current - velo_previous) / acc_tot(idx)


# Avståndet det tar att accelerera till den nya hastigheten
def dist_acc(idx):
    velo_previous = float(Current_pd(speed, indx(prev_point(idx)))) / 3.6
    # s = v0*t + (a*t^2)/2
    return velo_previous * time_acc(idx) + acc_tot(idx) * (time_acc(idx)**2) / 2


def time_constant_velo(idx):
    velo_current = float(Current_pd(speed, idx)) / 3.6

    #  s = v0*t
    #  *1000 för att omvandla till meter
    return ((float(dists[idx])*1000) - float(dist_acc(idx)))/velo_current


def dist_const_velo(idx):
    #  Totala sträckan till nästa punkt minus sträckan under acceleration
    return (float(dists[idx])*1000) - float(dist_acc(idx))


#  Energy consumption during constant velocity
def energy_acc(idx):
    velo_current = float(Current_pd(speed, idx)) / 3.6
    velo_previous = float(Current_pd(speed, indx(prev_point(idx)))) / 3.6

    #  Regenerative breaking
    if acc_tot(idx) >= 0:
        return force_traction_acc(idx) * dist_acc(idx) / eta
    #  Maximum deaceleration of 2.9 m/s^2
    elif -2.9 < acc_tot(idx) < 0:
        #  Stores the kinetic energy
        #  * eta since acceleration is negative
        return -(eta * m_bil * (velo_current - velo_previous) ** 2) / 2
    else:
        return 0


#  Energy consumption during acceleration
def energy_const_velo(idx):
    return force_traction_const(idx) * dist_const_velo(idx) / eta


#  Energy consumption due to heating in the cabin
def energy_hvch(idx):
    return HVCH_power/(eta_HVCH*(time_acc(idx)+time_constant_velo(idx))) #  osäker om tiden?


#  Total energy consumption
def total_energy(idx):
    #  Energy while accelerating + energy while constant velocity + energy due to heating
    tot_energy = ((energy_acc(idx) + energy_const_velo(idx)) * slip + energy_hvch(idx))

    return tot_energy


# Change in state of charge
def s_o_c_change(idx, soc):
    return total_energy(idx)/(u_o_c(soc)*200*3600)


# Open circuit voltage for the battery
def u_o_c(soc):
    #  KANSKE ÄNDRA DENNA FUNCTION
    return (soc * 5.83 * (10**-3) + 3.43)

def battery_temperature_change(idx, soc, battery_temperature, t_active_charger, time):
    #  (T2-T1)*cp*m = Qgen + Qexh + Qact

    R_e = (3*l_battery) / visc_air   # Re-number
    N_u = 0.664*R_e**(1/2)*prandtl**(1/3)    # Nu- number, flate plate, laminar flow
    h_conv = (N_u*k_air)/l_battery           # H-number
    t_active = cp_battery * mass_battery * battery_temperature / (HVCH_power * eta_HVCH)

    #  Ahads equation but with number of cells is series and divided by time
    Q_loss = internal_resistance_battery(battery_temperature)*((total_energy(idx)/(u_o_c(soc)*(time_acc(idx)+time_constant_velo(idx))*cells_in_series))**2)*(time_acc(idx)+time_constant_velo(idx))
    Q_exchange = 2*h_conv*l_battery*w_battery*(battery_temperature-t_amb)
    Q_drive = abs(0.03*(energy_acc(idx)+energy_const_velo(idx))) 
    Q_cooling = HVCH_power*(time_acc(idx) + time_constant_velo(idx))*eta_HVCH

    if battery_temperature > 273+35:
        d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive - Q_cooling)
    elif battery_temperature > 273+15 and time < t_active_charger:
        d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive - Q_cooling)
    else:
        d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive)

    

    return d_T, t_active


def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000



def iterate(idx_start: int, route: int, soc: float, batt_temp: float, t_active_charger: float) -> tuple:
    """ Func that iterates through the index-points. Calculates the energy consumed, distance moved, timechange, SOC and 
    which charging-stations are reachable while not running out of energy (soc<20)"""

    init_df(route)

    len_df = len(df)

    # Starting values
    battery_temperature = batt_temp  # Starting with ambient temp of air
    total_energy_consumption = 0
    temp_iter = 0
    total_distance = 0
    total_time = 0
    plot_idx = 0
    charge_dict = {}   
    plot_parameters = \
    {
        'idx': [],
        'temp': [],
        'dist': [],
        'time': []
    }
    first_trial = True
    prev_soc = 80
    index = idx_start

    #  Iterating over road points
    while index < len_df-1:
        
        # For every iteration calculate and add each import
        
        prev_values = \
        {
            'energy_con': total_energy_consumption, 
            'soc': soc, 
            'distance': total_distance, 
            'time': total_time, 
            'temp': battery_temperature,
            'index': index,
            'temp_iter': temp_iter,
            'plot_index': plot_idx,
            'plot_params': plot_parameters
        }


        
        total_energy_consumption += total_energy(index)
        soc -= s_o_c_change(index, soc)
        total_distance += (dist_const_velo(index) + dist_acc(index))
        total_time += (time_acc(index) + time_constant_velo(index))
        dT, t_active = battery_temperature_change(index, soc, battery_temperature, t_active_charger, total_time)
        battery_temperature += dT
        temp_iter += dT
        
        plot_parameters['temp'].append(battery_temperature)
        plot_parameters['dist'].append(total_distance/1000)
        plot_parameters['time'].append(total_time/60)
        plot_parameters['idx'].append(index)
        params = \
        {
            'energy_con': total_energy_consumption, 
            'soc': soc, 
            'distance': total_distance, 
            'time': total_time, 
            'temp': battery_temperature,
            'index': index,
            'temp_iter': temp_iter,
            'plot_index': plot_idx,
            'plot_params': plot_parameters
        }



        # Total energy in battery: 75kWh * 3600 * 1000 joules = 270 000 kJ

        # If Soc is less than 40 procent, look for chargers
        if 20 < soc < 40:
            # Kollar nu bara punkter efter soc=40, men inte alla laddare efter 40... 
            # Borde köra att funktionen kollar laddare efter idx-1 när vi når soc<40
    
            try:
                # Grabs chargers associated with data points
                chargers = tuple(map(str, Current_pd(df, index)['next chargers'].split(', ')))
            except:
                pass

            # First time we get soc < 40, we collect all the chargers that were in the previous interval
            if first_trial:
                try:
                    # Grabs chargers associated with data points
                    chargers_prev = tuple(map(str, Current_pd(df, index-1)['next chargers'].split(', ')))
                    for charger in chargers_prev:
                        if charger != "0":
                            #charge_dict[charger] = (soc, total_time, index, total_energy_consumption, total_distance, battery_temperature, road_2)
                            charge_dict[charger] = prev_values
                except:
                    pass
                first_trial = False
            
            # If there is a charger close, save it
            for charger in chargers:
                if charger != "0":
                    #charge_dict[charger] = (soc, total_time, index, total_energy_consumption, total_distance, battery_temperature, road_2)
                    charge_dict[charger] = \
                    {
                        'energy_con': total_energy_consumption, 
                        'soc': soc, 
                        'distance': total_distance, 
                        'time': total_time, 
                        'temp': battery_temperature,
                        'index': index,
                        'temp_iter': temp_iter,
                        'plot_index': plot_idx,
                        'plot_params': plot_parameters
                        
                    }
                    
        elif soc < 20:
            # charge_dict = iterate_charger(charge_dict, battery_temperature, soc, index)    "" ""
            charge_dict_new = iterate_charger(charge_dict, route)
            print(charge_dict_new)
            ##soc = 80    # nyladdat batteri
            return charge_dict_new, False

        index += 1
        plot_idx += 1
    return params, True


######################
## Hjälp funktioner ##
######################


# Hittar nästa punkt från nuvarnde idx
def next_point(idx):
    return df.iloc[idx]['next']


# Hittar tidigare punkt från nuvarnde idx
def prev_point(idx):
    return df.iloc[idx]['previous']


# Hämtar index för en punkt, givet en column och ett namn
def indx(name):
    return list(df['name']).index(name)


# Tar ut värdet för en column, givet en punkt
def Current_pd(pds, idx):
    return pds.iloc[idx]

