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
    global df, df_charge, grads, speed, dists, dist_to_point_after_charger, charger_elev, nearest_highway, highway_dist
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
    nearest_highway = df_charge['nearest highway']
    highway_dist = df_charge['nearest highway']

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
    elev_current = point["current_elev"]
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
def s_o_c_change(point, soc):
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
    print(Q_exchange, Q_loss, Q_drive, h_conv)

    d_T = (1/(cp_battery*mass_battery))*(-Q_exchange + Q_loss + Q_drive)

    return d_T


def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000


def iterate_charger(chargers: dict, temp: int, s_o_c: int, start_idx: int) -> dict:

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
        """
        battery_temperature = temp
        total_energy_consumption = 0
        total_distance = 0
        total_time = 0
        soc = s_o_c
        """
        soc = charger[0]
        total_time = charger[1]
        start_idx = charger[2]
        total_energy_consumption = charger[3]
        total_distance = charger[4]
        battery_temperature = charger[5]

        charger_idx = df_charge['name'].index(charger)
        
        #TEST POINT for a charger going from location where we "decide" to charge.
        #This should be changed so we have 4 points (represented as 3 "point" since it contains info about both of them).
        # 1. where we decide to charge
        # 2. where we leave the highway
        # 3. where the charger is 
        # 4. where we go back onto the highway
        point = {
            "current_velocity": charger_speed_limit,
            "prev_velocity": float(Current_pd(speed, start_idx)),
            "current_elev": charger_elev[charger_idx],
            "dist_to_next_point": dist_to_point_after_charger[charger_idx],
        }

        # 1: start_idx -> nearest_highway (räkna ut dist som kommer vara starting distance (total_distance))
        # 2: nearest_highway -> charger (finns "dist to highway")
        # 3: charger -> nearest_highway (finns "dist to highway")

        highway_point = nearest_highway[charger_idx]
        higway_speed_limit = float(Current_pd(speed, start_idx))

        p1 = {
            "current_velocity": float(Current_pd(speed, start_idx)), # highway_speed_limit,         # uppdaterade hastigheterna tilll det de borde vara enligt dina instruktioner / henrik
            "prev_velocity": float(Current_pd(speed, prev_point(start_idx))),
            "current_elev": 0, #finns det någon för highway? annars ta samma som start_idx
            "dist_to_next_point": highway_dist[charger_idx] #dist to charger
        }

        p2 = {
            "current_velocity": charger_speed_limit,
            "prev_velocity": float(Current_pd(speed, start_idx)), # highway_speed_limit,
            "current_elev": charger_elev[charger_idx],
            "dist_to_next_point": highway_dist[charger_idx] #dist back to highway
        }

        p3 = {
            "current_velocity": charger_speed_limit, # highway_speed_limit,
            "prev_velocity": charger_speed_limit,
            "current_elev": 0, #finns det någon för highway? annars ta samma som start_idx
            "dist_to_next_point": 0 #TODO: Här behöver vi till nästa punkt efter laddning.. detta måste komma in i Fordonsdynamik på något sätt
        }

        #Kanske kommer behöva något av detta senare eller så sker det i fordonsdynamik
        '''
        #get lat/long from start_index and nearest highway

        R = 6373.0

        #negativ longitude för jag tror alla longituder är W??

        lat1 = radians(lat1)
        lon1 = radians(-lon1)
        lat2 = radians(lat2)
        lon2 = radians(-lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        '''

        for _ in range(2):
            # For every iteration calculate and add each import
            temp_total_energy_consumption = total_energy_consumption + total_energy(point) 
            soc -= s_o_c_change(point, soc)
            total_distance += (dist_const_velo(point) + dist_acc(point))
            total_time += (time_acc(point) + time_constant_velo(point))         # [s]
            battery_temperature += battery_temperature_change(point, soc, battery_temperature)
        

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

# Hittar tidigare punkt från nuvarnde idx
def prev_point(idx):
    return df.iloc[idx]['previous']