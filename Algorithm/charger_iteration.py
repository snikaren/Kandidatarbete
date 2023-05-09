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
    global df, df_charge, grads, speed, dists, dist_to_point_after_charger, charger_grads_next, charger_grads_prev, nearest_highway, highway_dist
    global dist_from_highway_to_prev, dist_from_highway_to_next
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
    charger_grads_next = df_charge['gradient next']
    charger_grads_prev = df_charge['gradient prev']
    nearest_highway = df_charge['nearest highway']
    highway_dist = df_charge['dist to highway']

    dist_from_highway_to_prev = df_charge["dist to prev point"]
    dist_from_highway_to_next = df_charge["dist to next point"]

#  Läser in csv-filen # Dict med hastighet som key och acc_tid som value
data = np.loadtxt(r'Algorithm\excel\AccelerationsTider.csv', delimiter=";", skiprows=1)
acc_dict = dict()
for i in range(len(data)):
    acc_dict[data[i, 0]] = data[i, 1]

# Resulting acceleration of car
def acc_tot(point):
    velo_current = point["current_velocity"]
    velo_previous = point["prev_velocity"]

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return abs(velo_current - velo_previous)/(3.6*(acc_dict[velo_current] - acc_dict[velo_previous]))


def force_tot(point):
    velo_current = point["current_velocity"]
    velo_previous = point["prev_velocity"]

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        #  integrerar V/t = a, i vårt fall nästa punkt minus föregående
        return m_bil*(velo_current - velo_previous) / (3.6 * (acc_dict[velo_current] - acc_dict[velo_previous]))


# Dragforce
def force_air(point, mode):
    #  /3.6 omvandlar till m/s
    velo_current = point["current_velocity"] / 3.6
    velo_previous = point["prev_velocity"] / 3.6
    if mode == 0:
        #  Mean of final velocity and starting velocity
        return den_air * cd * area_front * ((velo_current + velo_previous) / 2) ** 2 / 2
    elif mode == 1:
        #  Final velocity
        return den_air * cd * area_front * velo_current ** 2 / 2


def force_elevation(point):
    elev_current = point["current_gradient"]
    return m_bil * g * (np.sin(elev_current) + c_r * np.cos(elev_current))


def force_traction_acc(point):
    return force_tot(point) + force_air(point, 0) + force_elevation(point)


def force_traction_const(point):
    return force_air(point, 1) + force_elevation(point)


## Beräkningar av effekt och arbete från Acc ##
# Beräknar tiden det tar att accelerera mellan två olika hastigheter
def time_acc(point):
    #  /3.6 omvandlar till m/s
    velo_current = point["current_velocity"] / 3.6
    velo_previous = point["prev_velocity"] / 3.6

    #  To handle the case where the next velocity is the same as the previous
    if velo_current == velo_previous:
        return 0

    else:
        return abs(velo_current - velo_previous) / acc_tot(point)


# Avståndet det tar att accelerera till den nya hastigheten
def dist_acc(point):
    #  /3.6 omvandlar till m/s
    velo_previous = point["prev_velocity"] / 3.6
    # s = v0*t + (a*t^2)/2
    return velo_previous * time_acc(point) + acc_tot(point) * (time_acc(point)**2) / 2


def time_constant_velo(point):
    #  /3.6 omvandlar till m/s
    velo_current = point["current_velocity"] / 3.6

    #  s = v0*t
    #  *1000 för att omvandla till meter
    return ((float(point["dist_to_next_point"])*1000) - float(dist_acc(point)))/velo_current


def dist_const_velo(point):
    #  Totala sträckan till nästa punkt minus sträckan under acceleration
    return (float(point["dist_to_next_point"])*1000) - float(dist_acc(point))


def energy_acc(point):
    "Energy consumption during accelertaion"
    
    velo_current = point["current_velocity"] / 3.6
    velo_previous = point["prev_velocity"] / 3.6

    #  Regenerative breaking
    if acc_tot(point)  >= 0:
        return force_traction_acc(point) * dist_acc(point) / eta
    #  Maximum deaceleration of 2.9 m/s^2
    elif -2.9 < acc_tot(point) < 0:
        #  Stores the kinetic energy
        #  * eta since acceleration is negative
        return -(eta * m_bil * (velo_current - velo_previous) ** 2) / 2
    else:
        return 0


#  Energy consumption during acceleration
def energy_const_velo(point):
    return force_traction_const(point) * dist_const_velo(point) / eta

#  Energy consumption due to heating in the cabin
def energy_hvch(point):
    return HVCH_power/(eta_HVCH*(time_acc(point)+time_constant_velo(point))) #  osäker om tiden?

#  Total energy consumption
def total_energy(point):
    #  Energy while accelerating + energy while constant velocity + energy due to heating
    tot_energy = ((energy_acc(point) + energy_const_velo(point)) * slip + energy_hvch(point))
    
    return tot_energy

# Change in state of charge
def s_o_c_change(point: dict, soc: float):
    return total_energy(point)/(u_o_c(soc)*200*3600)


# Open circuit voltage for the battery
def u_o_c(soc):
    #  KANSKE ÄNDRA DENNA FUNCTION
    return (soc * 5.83 * (10**-3) + 3.43)


def battery_temperature_change(point, soc, battery_temperature):
    #  (T2-T1)*cp*m = Qgen + Qexh + Qact

    R_e = (den_air*2*l_battery) / visc_air   # Re-number
    N_u = 0.664*R_e**(1/2)*prandtl**(1/3)    # Nu- number, flate plate, laminar flow
    h_conv = (N_u*k_air)/l_battery             # H-number

    #  Ahads equation but with number of cells is series and divided by time
    Q_loss = internal_resistance_battery(battery_temperature)*((total_energy(point)/(u_o_c(soc)*(time_acc(point)+time_constant_velo(point))*cells_in_series))**2)
    Q_exchange = h_conv*l_battery*w_battery*(battery_temperature-t_amb)
    Q_drive = abs(0.05*(energy_acc(point)+energy_const_velo(point)))

    d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive)

    return d_T


def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000


def iterate_charger(chargers: dict, route: int) -> dict:
    init_df(route)

    charge_dict = {}
    soc_at_charger = 0

    #  Iterating over road points
    for name, charger in chargers.items():
        charger_dist = 0
        soc = charger['soc']
        total_time = charger['time']
        start_idx = charger['index']
        total_energy_consumption = charger['energy_con']
        total_distance = charger['distance']
        battery_temperature = charger['temp']
        batt_temp_at_charger = 0

        charger_idx = df_charge.index[df_charge['name']==name].tolist()[0]

        # 1: start_idx -> nearest_highway (räkna ut dist som kommer vara starting distance (total_distance))
        # 2: nearest_highway -> charger (finns "dist to highway")
        # 3: charger -> nearest_highway (finns "dist to highway")


        p0 = {
            "current_velocity": float(Current_pd(speed, start_idx)), # highway_speed_limit,         
            "prev_velocity": float(Current_pd(speed, start_idx-1)),
            "current_gradient": float(Current_pd(charger_grads_prev, charger_idx)),  #finns det någon för highway? annars ta samma som start_idx
            "dist_to_next_point": float(Current_pd(dist_from_highway_to_prev, charger_idx)) #dist to charger
        }

        p1 = {
            "current_velocity": charger_speed_limit,
            "prev_velocity": float(Current_pd(speed, start_idx)), # highway_speed_limit,
            "current_gradient": -float(Current_pd(grads, start_idx)), #Vi kör åt andra hållet så bör vara negativt?
            "dist_to_next_point": float(Current_pd(highway_dist, charger_idx)) #dist back to highway
        }

        p2 = {
            "current_velocity": charger_speed_limit, # highway_speed_limit,
            "prev_velocity": 0,
            "current_gradient": float(Current_pd(grads, start_idx)),  #finns det någon för highway? annars ta samma som på vägen
            "dist_to_next_point": float(Current_pd(highway_dist, charger_idx))
        }

        p3 = {
            "current_velocity": float(Current_pd(speed, start_idx)), # highway_speed_limit,
            "prev_velocity": charger_speed_limit,
            "current_gradient": float(Current_pd(charger_grads_next, charger_idx)), #finns det någon för highway? annars ta samma som på vägen
            "dist_to_next_point": float(Current_pd(dist_from_highway_to_next, charger_idx))
        }
        total_distance = dist_from_highway_to_prev[charger_idx] #Starting distance

        for p in [p0, p1, p2, p3]:
            total_energy_consumption = total_energy_consumption + total_energy(p) 
            if p == p2:
                soc_at_charger = soc
                soc = 80
                batt_temp_at_charger = battery_temperature
            soc -= s_o_c_change(p, soc)
            total_distance += (dist_const_velo(p) + dist_acc(p))
            charger_dist += (dist_const_velo(p) + dist_acc(p))
            total_time += (time_acc(p) + time_constant_velo(p))
            battery_temperature += battery_temperature_change(p, soc, battery_temperature)

        if soc_at_charger > 20:
            charge_dict[name] = \
            {
                'energy_con': total_energy_consumption, 
                'soc_charger': soc_at_charger,
                'soc': soc,
                'distance': total_distance, 
                'time': total_time, 
                'temp': battery_temperature,
                'temp_at_charger': batt_temp_at_charger,
                'index': start_idx + 1,
                'highway_dist': charger_dist
            }

    return charge_dict


######################
## Hjälp funktioner ##
######################

# Tar ut värdet för en column, givet en punkt
def Current_pd(pds, idx):
    return pds.iloc[idx]

# Hittar tidigare punkt från nuvarnde idx
def prev_point(idx):
    return df.iloc[idx]['previous']

if __name__ == "__main__":
    init_df(1)
    x = iterate_charger({"jqne9e": (50, 0, 4, 0, 0, 274)}, 1)
    print(x)