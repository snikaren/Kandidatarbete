from Konstanter import *
import numpy as np
import pandas as pd
import time
import math as m
import matplotlib.pyplot as plt

# Dataframe uppsättning
df = ""

def init_df(route):
    global df
    if route == 1:
        df = pd.read_csv(r'Algorithm\rutt_1.csv')
    elif route == 2:
        df = pd.read_csv(r'Algorithm\rutt_2.csv')
    else:
        df = pd.read_csv(r'Algorithm\rutt_3.csv')

grads = df['gradient to next']
speed = df['speed limit']
dists = df['dist to next[Km]']
df_charge = pd.read_csv(r'C:\Users\Albert\OneDrive\Skrivbord\Fordonsdynamik_python\Fordonsdynamik\chargers.csv')


#  Läser in csv-filen
data = np.loadtxt('AccelerationsTider.csv', delimiter=";", skiprows=1)

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
    if force_tot(idx) >= 0:
        return force_traction_acc(idx) * dist_acc(idx) / eta
    #  Maximum deaceleration of 2.9 m/s^2
    elif -2.9 < force_tot(idx) < 0:
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


def battery_temperature_change(idx, soc, battery_temperature):
    #  (T2-T1)*cp*m = Qgen + Qexh + Qact

    R_e = (den_air*2*l_battery) / visc_air   # Re-number
    N_u = 0.664*R_e**(1/2)*prandtl**(1/3)    # Nu- number, flate plate, laminar flow
    h_conv = (N_u*k_air)/l_battery             # H-number

    #  Ahads equation but with number of cells is series and divided by time
    Q_loss = internal_resistance_battery(battery_temperature)*((total_energy(idx)/(u_o_c(soc)*(time_acc(idx)+time_constant_velo(idx))*cells_in_series))**2)
    Q_exchange = h_conv*l_battery*w_battery*(battery_temperature-t_amb)
    Q_drive = abs(0.05*(energy_acc(idx)+energy_const_velo(idx)))
    print(Q_exchange, Q_loss, Q_drive, h_conv)

    d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive)

    return d_T


def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000



def iterate(idx_start, route):

    # dataframe
    init_df(route)

    # Starting values
    battery_temperature = 274  # Starting with ambient temp of air
    total_energy_consumption = 0
    total_distance = 0
    total_time = 0
    soc = 100 * max_battery/Battery_joules  # Starting with soc=80

    #  Lists to append to
    indices = [0]
    current_energy_consumption_list = [0]
    total_energy_consumption_list = [0]
    s_o_c_list = [80]
    u_o_c_list = [0]
    battery_temperature_list = [274]
    total_distance_list = [0]
    internal_resistance_battery_list = []

    start_time = time.time()
    charge_dict = {}
    #  Iterating over road points
    for index in range(idx_start, len(df)-1):

        # For every iteration calculate and add each import
        total_energy_consumption += total_energy(index)
        soc -= s_o_c_change(index, soc)
        total_distance += (dist_const_velo(index) + dist_acc(index))
        total_time += (time_acc(index) + time_constant_velo(index))         # [s]
        battery_temperature += battery_temperature_change(index, soc, battery_temperature)
        #print(battery_temperature_change(index, soc, battery_temperature))

        # Total energy in battery: 75kWh * 3600 * 1000 joules = 270 000 kJ

        # If Soc is less than 20 procent, look for chargers
        if 20 < soc < 40:

            # Grabs chargers associated with data points
            chargers = tuple(map(str, Current_pd(df, index)['next chargers'].split(', ')))
            
            # If there is a charger close, save it
            for charger in chargers:
                if charger != "0":
                    charge_dict[charger] = (soc, total_time)
            """
            #  Adds second identical data point when charging
            indices.append(index)
            current_energy_consumption_list.append(total_energy(index))
            total_energy_consumption_list.append(total_energy_consumption)
            s_o_c_list.append(soc)
            u_o_c_list.append(u_o_c(soc))
            battery_temperature_list.append(battery_temperature)
            total_distance_list.append(total_distance)

            print(f"Charge before point: {index-1}")
            print(soc)

            soc = 80
            """

        elif soc < 20:
            charge_dict = iterate_charger(charge_dict) # NOT IMPLEMENTED
            return charge_dict, True
        
    

        """ Hämtar info för Plottar """
        indices.append(index)
        current_energy_consumption_list.append(total_energy(index))
        total_energy_consumption_list.append(total_energy_consumption)
        s_o_c_list.append(soc)
        u_o_c_list.append(u_o_c(soc))
        battery_temperature_list.append(battery_temperature)
        total_distance_list.append(total_distance)
    
    print(charge_dict)
    print(battery_temperature_list)

    end_time = time.time()

    fig, axes = plt.subplots(2, 2)
    axes[0][0].plot(indices, current_energy_consumption_list, 'r')
    axes[0][0].set_title('Current energy consumption over data points', fontsize=7)
    axes[0][0].set_xlabel('Indices of data points', fontsize=5)
    axes[0][0].set_ylabel('Energy consumption [MJ]', fontsize=5)
    axes[0][0].set_xticks(indices)
    axes[0][0].tick_params(axis='x', labelsize=4)
    axes[0][0].set_yticklabels(['0', '2', '4', '6', '8', '10', '12', '14'], fontsize=6)

    axes[0][1].plot(indices, total_energy_consumption_list, 'g')
    axes[0][1].set_title('Total energy consumption over data points', fontsize=7)
    axes[0][1].set_xlabel('Indices of data points', fontsize=5)
    axes[0][1].set_ylabel('Energy consumption [J]', fontsize=5)
    axes[0][1].set_xticks(indices)
    axes[0][1].tick_params(axis='x', labelsize=4)
    #axes[0][1].set_yticklabels(['0', '50', '100', '150', '200', '250', '300','350', '400', '450', '500', '550', '600', '650', '700'], fontsize=6)

    axes[1][1].plot(indices, s_o_c_list, 'b')
    axes[1][1].set_title('State of charge over data points', fontsize=7)
    axes[1][1].set_xlabel('Indices of data points', fontsize=5)
    axes[1][1].set_ylabel('State of charge [%]', fontsize=5)
    axes[1][1].set_xticks(indices)
    #axes[1][1].set_yticklabels(['0', '20', '40', '80', '100'], fontsize=6)
    axes[1][1].tick_params(axis='x', labelsize=4)

    axes[1][0].plot(total_distance_list, s_o_c_list)
    axes[1][0].set_title('State of charge over distance', fontsize=7)
    axes[1][0].set_xlabel('Distance [km]', fontsize=5)
    axes[1][0].set_ylabel('State of charge [%]', fontsize=5)
    #axes[1][0].set_yticklabels(['0', '20', '40', '80', '100'], fontsize=6)
    #axes[1][0].set_xticks(soc)
    axes[1][0].tick_params(axis='x', labelsize=4)
    axes[1][0].set_xticklabels(['0', '100', '200', '300', '400', '500'], fontsize=6)

    plt.tight_layout()

    fig, axess = plt.subplots(2, 2)
    axess[0][0].plot(total_distance_list, battery_temperature_list)
    axess[0][0].set_title('Battery temperature over distance', fontsize=7)
    axess[0][0].set_xlabel('Distance [km]', fontsize=5)
    axess[0][0].set_ylabel('Battery temperature [C]', fontsize=5)
    # axes[1][0].set_yticklabels(['0', '20', '40', '80', '100'], fontsize=6)
    # axes[1][0].set_xticks(soc)
    axess[0][0].tick_params(axis='x', labelsize=4)
    axess[0][0].set_xticklabels(['0', '100', '200', '300', '400', '500'], fontsize=6)

    # axes[1][0].plot(s_o_c_list, u_o_c_list)
    # axes[1][0].set_title('Open circuit voltage over state of charge', fontsize=7)
    # axes[1][0].set_xlabel('State of charge [%]', fontsize=5)
    # axes[1][0].set_ylabel('Open circuit voltage [V]', fontsize=5)
    # #axes[1][0].set_xticks(soc)
    # axes[1][0].tick_params(axis='x', labelsize=4)

    #print(f"Time computing: {end_time-start_time}")
    #print(f"Total energy consumption: {round(total_energy_consumption / 1000, 4)} [kJ]")
    #print(f"State of charge: {round(soc, 2)} %")
    #print(f"Total distance: {round(total_distance) / 1000} [km]")
    #print(f"Total time: {round(total_time / 3600, 2)} [h]")

    plt.show()

    return charge_dict, False


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

