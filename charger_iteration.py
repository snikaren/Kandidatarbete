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


def iterate_charger(chargers):

    # Starting values
    battery_temperature = 274  # Starting with ambient temp of air
    total_energy_consumption = 0
    total_distance = 0
    total_time = 0
    soc = 100 * max_battery/Battery_joules  # Starting with soc=80

    start_time = time.time()
    #  Iterating over road points
    for charger in chargers:
        for index in [0,1]:
            # For every iteration calculate and add each import
            temp_total_energy_consumption = total_energy_consumption + total_energy(index)
            soc -= s_o_c_change(index, soc)
            total_distance += (dist_const_velo(index) + dist_acc(index))
            total_time += (time_acc(index) + time_constant_velo(index))         # [s]
            battery_temperature += battery_temperature_change(index, soc, battery_temperature)
    return