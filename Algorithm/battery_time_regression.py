import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pandas as pd
import csv
import random
import math as m

"""
Stödfunktioner: energiförbrukning laddning

"""
#CSV
df = pd.read_csv(r'Algorithm\excel\Charging_curves.csv',delimiter=";")

def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000

def u_o_c(soc):
    return (soc * 5.83 * (10**-3) + 3.43)

def charging_powah():
    " Get capacity from gatherd data over several EV"
    charging_powah = [] #Lista med EV:s kapacitet
    for i in range(1,len(df.iloc[0])):
        charging_powah.append(df.iloc[0,i])
    return charging_powah

def time(init_soc,final_soc,charging_powah,soc_value,charger,battery_temperature):
    den_air = 1.292   
    l_battery = 2.75
    visc_air = 1.338*10**(-5)
    prandtl = 0.7362 
    k_air = 0.02364   
    R_e = (den_air*2*l_battery) / visc_air 
    N_u = 0.664*R_e**(1/2)*prandtl**(1/3)    
    h_conv = (N_u*k_air)/l_battery             
    cells_in_series = 108
    w_battery = 1.5
    cp_battery = 900 
    mass_battery = 312
    t_amb = 273.15

    temp = {}
    Q_loss_total_dict= {}
    max_charging_power_dict = {}
    energy_dict = {}
    energy_list = []
    a = 0 #iterrera genom charging_powah
    for i in range(1,len(df.columns)):
        max_charging_power = df.iloc[11:72,i].max() #find the maximum charging power
        time_list = []   
        time = 0 #reset time
        battery_capacity = charging_powah[a]
        init_soc = soc_value
        Q_loss_total = 0
        battery_temp = battery_temperature
        for soc in range(init_soc,final_soc):
            delta_soc = min(1,final_soc-init_soc)
            row_index = soc - 9  
            charging_power = df.iloc[row_index, i]
          
            #Energy
            init_energy = battery_capacity * (init_soc)/100 #init_energy = battery_capacity * (init_soc/100)
            final_energy = battery_capacity * ((soc+delta_soc)/100) #final_energy = battery_capacity * ((soc+delta_soc)/100)
            energy = final_energy-init_energy #since delta_soc = 1 always this will be the battery capacity/100

            Charging_time = (energy / charging_power) #assume 100% efficiency

            energy_cons = Charging_time * charging_power

            #charge_provided = {20:100, 21: 87, 22:78 ...}
                        
            Q_loss = internal_resistance_battery(battery_temp)*(((charging_power*1000)/(u_o_c(i)*cells_in_series))**2)*(Charging_time*3600) #Q loss i Joule
            Q_loss_total += internal_resistance_battery(battery_temp)*(((charging_power*1000)/(u_o_c(i)*cells_in_series))**2)*(Charging_time*3600)
            #Q_exchange = h_conv*l_battery*w_battery*(battery_temperature-t_amb)
            d_T = (Q_loss/(cp_battery*mass_battery)) 
            battery_temp += d_T

            time += Charging_time

            time_list.append(time)
            #soc_list.append(init_soc)

            init_soc += delta_soc
        temp[max_charging_power] = battery_temp-273.15 #Tillbaka till celcius
        Q_loss_total_dict[max_charging_power] = Q_loss_total * 0.000000277778 #tillbaka till kwh

        energy_list = [energy_cons for _ in range((final_soc-soc_value-1))]
        energy_dict[battery_capacity] = sum(energy_list)
        max_charging_power_dict[max_charging_power] = (time_list) #Dictionary with key:Max charging power -> time list for EV
        a +=1
    

    x_data_1 = list(Q_loss_total_dict.keys())
    x_data_1.pop(-1)
    x_data_1 = np.array(x_data_1).reshape((-1,1))
    y_data_energy = list(Q_loss_total_dict.values())
    y_data_energy.pop(-1)
    y_data_energy = np.array(y_data_energy)
    y_data_temp = list(temp.values())
    y_data_temp.pop(-1)
    y_data_temp = np.array(y_data_temp)

    # Create a linear regression model and fit the data
    model_1 = LinearRegression()
    model_2 = LinearRegression()
    model_1.fit(x_data_1, y_data_energy)
    model_2.fit(x_data_1,y_data_temp)

    # Predict y-values using the model
    predicted_energy = model_1.predict(np.array([[charger]])) + ((final_soc-soc_value)/100)*69 #Lägger till Energi för laddningen
    predicted_energy = predicted_energy*3,6*10^6 #Gör om till Joule
    predicted_temp = model_2.predict(np.array([[charger]]))


    #return("predicted energy:",predicted_energy,"predicted_temp",predicted_temp)
    #y_pred_x=model.predict(x_data_1)
    


    #return (temp,"Q_loss=",Q_loss_total_dict)

    pre_header_array = list(max_charging_power_dict)
    header_array = ["SOC"] + pre_header_array
    SOC_array = np.arange(soc_value,80,1) #array 20 -> 80

    with open("Algorithm/excel/time_real.csv","w",newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header_array)
        for i in range(len(SOC_array)):
            row = [SOC_array[i]]
            for key in max_charging_power_dict.keys():
                row.append(max_charging_power_dict[key][i])
            writer.writerow(row)
    #uppdaterad
    df2 = pd.read_csv(r"Algorithm/excel/time_real.csv",delimiter=",")
    x_data = list(max_charging_power_dict.keys()) #samtliga charging power lista
    x_data.pop(-1)
    y_data = list(df2.iloc[-1,1:].values)#lista med tot tid att ladda
    y_data.pop(-1)

    t = 0
    for value in y_data:
        y_data[t] = (69/charging_powah[t]) * value #Få den riktiga totala tiden med 69 kWh batteri
        t+=1
    
    data = list(zip(x_data, y_data))
    charge_power, time_to_charge = list(zip(*data))
    charge_power = list(charge_power)
    time_to_charge = list(time_to_charge)

    p = np.polyfit(charge_power, time_to_charge,5)
    return(predicted_energy,np.polyval(p,charger),predicted_temp)

def main(soc: float, cap: int, charging_powah, battery_temp) -> tuple:



    #Parametrar
    
    init_soc = int(soc) #int(input("Enter initial state of charge: "))
    soc_value = init_soc
    final_soc = 80
    charger = cap # int(input("Enter the charger power provided: "))
    if charger > 150:
        charger = 150

    batteritemp = battery_temp

    #Spotta ut tiden
    time_to_charge = time(init_soc,final_soc,charging_powah,soc_value,charger,batteritemp)
    return time_to_charge
    #print("the time it takes to charge is",time_to_charge,"and the charge provided is", charge_provided)

    #Ge energiförlust i kwH
    #FL=energy_function(batteritemp,time_to_charge,charge_provided)
    #print(FL)



if __name__ == "__main__":
    main()

