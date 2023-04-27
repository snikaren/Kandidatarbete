from Konstanter import *
import numpy as np
import pandas as pd
import math as m


charger_speed_limit = 50
# Dataframe uppsättning
df = ""
grads, speed, dists = 0,0,0
df_charge = pd.read_csv(r'Algorithm\excel\chargers.csv')

def init_df(route):
    global df, df_charge, grads, speed, dists, dist_to_point_after_charger, charger_elev
    if route == 1:
        df = pd.read_csv(r'Algorithm\excel\rutt_1.csv')
    elif route == 2:
        df = pd.read_csv(r'Algorithm\excel\rutt_2.csv')
    else:
        df = pd.read_csv(r'Algorithm\excel\rutt_3.csv')

    grads = df['gradient to next']
    speed = df['speed limit']
    dists = df['dist to next[Km]']
    dist_to_point_after_charger = df_charge['dist to next point']
    charger_elev = df_charge['elevation']

#  Läser in csv-filen # Dict med hastighet som key och acc_tid som value
data = np.loadtxt(r'Algorithm\excel\AccelerationsTider.csv', delimiter=";", skiprows=1)
acc_dict = dict()
for i in range(len(data)):
    acc_dict[data[i, 0]] = data[i, 1]

# Resulting acceleration of car
def acc_tot(prev_idx):
    velo_current = charger_speed_limit
    velo_previous = float(Current_pd(speed, prev_idx))

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return abs(velo_current - velo_previous)/(3.6*(acc_dict[velo_current] - acc_dict[velo_previous]))


def force_tot(prev_idx):
    velo_current = charger_speed_limit
    velo_previous = float(Current_pd(speed, prev_idx))

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return m_bil*(velo_current - velo_previous) / (3.6 * (acc_dict[velo_current] - acc_dict[velo_previous]))


# Dragforce
def force_air(prev_idx, mode):
    velo_current = charger_speed_limit / 3.6
    velo_previous = float(Current_pd(speed, prev_idx)) / 3.6
    if mode == 0:
        #  Mean of final velocity and starting velocity
        return den_air * cd * area_front * ((velo_current + velo_previous) / 2) ** 2 / 2
    elif mode == 1:
        #  Final velocity
        return den_air * cd * area_front * velo_current ** 2 / 2


def force_elevation(charger_idx):
    elev_current = charger_elev[charger_idx]
    return m_bil * g * (np.sin(elev_current) + c_r * np.cos(elev_current))


def force_traction_acc(prev_idx, charger_idx):
    return force_tot(prev_idx) + force_air(prev_idx, 0) + force_elevation(charger_idx)


def force_traction_const(prev_idx, charger_idx):
    return force_air(prev_idx, 1) + force_elevation(charger_idx)


## Beräkningar av effekt och arbete från Acc ##
# Beräknar tiden det tar att accelerera mellan två olika hastigheter
def time_acc(prev_idx):
    #  /3.6 omvandlar till m/s
    velo_current = charger_speed_limit / 3.6
    velo_previous = float(Current_pd(speed, prev_idx))/3.6

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        return abs(velo_current - velo_previous) / acc_tot(prev_idx)


# Avståndet det tar att accelerera till den nya hastigheten
def dist_acc(prev_idx):
    velo_previous = float(Current_pd(speed, prev_idx)) / 3.6
    # s = v0*t + (a*t^2)/2
    return velo_previous * time_acc(prev_idx) + acc_tot(prev_idx) * (time_acc(prev_idx)**2) / 2


def time_constant_velo(prev_idx, charger_idx):
    velo_current = charger_speed_limit / 3.6

    #  s = v0*t
    #  *1000 för att omvandla till meter
    return ((float(dist_to_point_after_charger[charger_idx])*1000) - float(dist_acc(prev_idx)))/velo_current


def dist_const_velo(prev_idx, charger_idx):
    #  Totala sträckan till nästa punkt minus sträckan under acceleration
    return (float(dist_to_point_after_charger[charger_idx])*1000) - float(dist_acc(prev_idx))

#  Energy consumption during constant velocity
def energy_acc(prev_idx, charger_idx):
    velo_current = charger_speed_limit / 3.6
    velo_previous = float(Current_pd(speed, prev_idx)) / 3.6

    #  Regenerative breaking
    if force_tot(prev_idx) >= 0:
        return force_traction_acc(prev_idx, charger_idx) * dist_acc(prev_idx) / eta
    #  Maximum deaceleration of 2.9 m/s^2
    elif -2.9 < force_tot(prev_idx) < 0:
        #  Stores the kinetic energy
        #  * eta since acceleration is negative
        return -(eta * m_bil * (velo_current - velo_previous) ** 2) / 2
    else:
        return 0


#  Energy consumption during acceleration
def energy_const_velo(prev_idx, charger_idx):
    return force_traction_const(prev_idx, charger_idx) * dist_const_velo(prev_idx, charger_idx) / eta

#  Energy consumption due to heating in the cabin
def energy_hvch(prev_idx, charger_idx):
    return HVCH_power/(eta_HVCH*(time_acc(prev_idx)+time_constant_velo(prev_idx, charger_idx))) #  osäker om tiden?

#  Total energy consumption
def total_energy(prev_idx, charger_idx):
    #  Energy while accelerating + energy while constant velocity + energy due to heating
    tot_energy = ((energy_acc(prev_idx, charger_idx) + energy_const_velo(prev_idx, charger_idx)) * slip + energy_hvch(prev_idx, charger_idx))

    return tot_energy

# Change in state of charge
def s_o_c_change(prev_idx, charger_idx, soc):
    return total_energy(prev_idx, charger_idx)/(u_o_c(soc)*200*3600)


# Open circuit voltage for the battery
def u_o_c(soc):
    #  KANSKE ÄNDRA DENNA FUNCTION
    return (soc * 5.83 * (10**-3) + 3.43)


def battery_temperature_change(prev_idx, charger_idx, soc, battery_temperature):
    #  (T2-T1)*cp*m = Qgen + Qexh + Qact

    R_e = (den_air*2*l_battery) / visc_air   # Re-number
    N_u = 0.664*R_e**(1/2)*prandtl**(1/3)    # Nu- number, flate plate, laminar flow
    h_conv = (N_u*k_air)/l_battery             # H-number

    #  Ahads equation but with number of cells is series and divided by time
    Q_loss = internal_resistance_battery(battery_temperature)*((total_energy(prev_idx, charger_idx)/(u_o_c(soc)*(time_acc(prev_idx)+time_constant_velo(prev_idx, charger_idx))*cells_in_series))**2)
    Q_exchange = h_conv*l_battery*w_battery*(battery_temperature-t_amb)
    Q_drive = abs(0.05*(energy_acc(prev_idx, charger_idx)+energy_const_velo(prev_idx, charger_idx)))
    print(Q_exchange, Q_loss, Q_drive, h_conv)

    d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive)

    return d_T


def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000


def iterate_charger(chargers: dict, temp: int, s_o_c: int, start_idx: int):

    #For every charger
        # Get the start time
        # Set the initial values (temp, soc, etc)
        # Simulate from the start point to the charger
        # Get the end time
        # Calculate the total time
        # Calculate the total energy consumption
        # Calculate the total distance
    # Return modified chargers dict

    charge_dict = {}

    #  Iterating over road points
    for charger in chargers:
        battery_temperature = temp
        total_energy_consumption = 0
        total_distance = 0
        total_time = 0
        soc = s_o_c

        charger_idx = df_charge['name'].index(charger)

        # For every iteration calculate and add each import
        temp_total_energy_consumption = total_energy_consumption + total_energy(start_idx, charger_idx)
        soc -= s_o_c_change(start_idx, charger_idx, soc)
        total_distance += (dist_const_velo(start_idx, charger_idx) + dist_acc(start_idx))
        total_time += (time_acc(start_idx) + time_constant_velo(start_idx, charger_idx))         # [s]
        battery_temperature += battery_temperature_change(start_idx, charger_idx, soc, battery_temperature)

        charge_dict[charger] = \
        {
            'energy_con': temp_total_energy_consumption, 
            'soc': soc, 
            'distance': total_distance, 
            'time': total_time, 
            'temp': battery_temperature
        }

    return charge_dict


######################
## Hjälp funktioner ##
######################

# Tar ut värdet för en column, givet en punkt
def Current_pd(pds, idx):
    return pds.iloc[idx]