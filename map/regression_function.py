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
def internal_resistance_battery(battery_temperature):
    return (90.196*m.e**(-0.08*(battery_temperature-274)) + 25.166)/1000

def u_o_c(soc):
    return (soc * 5.83 * (10**-3) + 3.43)


"""
Energifunktion: energiförbrukning under laddning -> kapacitet + loss
"""

def energy_function(battery_temperature,tid,charge):
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

    #charge = {20:100, 21: 87, 22:78 ...}
    batteri = battery_temperature-273.15


    Q_loss = 0
    for i in charge:
        uoc = u_o_c(i)
        Q_loss = internal_resistance_battery(battery_temperature)*((charge[i]/(uoc*cells_in_series))**2)
        Q_exchange = h_conv*l_battery*w_battery*(battery_temperature-t_amb)

        d_T = ((1/(cp_battery*mass_battery))*(Q_loss-Q_exchange)) #-Q_exchange
        batteri += d_T

    E_charge = Q_loss*tid + Q_exchange

    return("Effektförlust:",E_charge,"W","ny batteritemp är:", batteri)

 #Q_loss += internal_resistance_battery(battery_temperature)*((total_energy(idx)/(u_o_c(SOC)*cells_in_series))**2)

"""
Skapandet av Charger power till EV för varje SOC beroende på vilken laddare (kW)  väljer 

"""

def charger_power_station(power,init_soc):
    df = pd.read_csv('Charging_curves.csv',delimiter=";")
    charger_power_provided = {}
    final_soc = 81
    #ex)power = 100 kW
    #Skapa dictionary med key: SOC, value: kW (average från Charging_curves.csv)
    
    #sök igenom samtliga max_charging power -> hitta intervall +-5 ? -> ta medelvärde
    for i in range(1,len(df.columns)):
        max_power = df.iloc[11:72,i].max()
        if power - 5 <=max_power <= power + 5: #kolla samtliga laddare som är i intervallet
            for soc in range(init_soc,final_soc):
                row_index = soc - 9  
                charging_power = df.iloc[row_index, i]
                if soc not in charger_power_provided:
                    charger_power_provided[soc] = []
                charger_power_provided[soc].append(charging_power)
    
    #skapa dictionary med value -> medelvärde Charging powah för varje soc
    for a in charger_power_provided.keys():
        list = charger_power_provided[a] #lista med alla kW [153,145,140 ..]
        mean = 0
        for i in list:
            mean += i
        mean_val = mean/(len(list))
        charger_power_provided[a] = mean_val
            
    return charger_power_provided


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
    df = pd.read_csv('Charging_curves.csv',delimiter=";")
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
                        
            Q_loss = internal_resistance_battery(battery_temp)*(((charging_power*1000)/(u_o_c(i)*cells_in_series))**2)*(Charging_time*3600) #Q loss i J
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
    predicted_energy = model_1.predict(np.array([[charger]])) #+ ((final_soc-soc_value)/100)*69 #Lägger till Energi för laddningen
    #predicted_energy = predicted_energy*3,6*10^6 #Gör om till Joule
    predicted_temp = model_2.predict(np.array([[charger]]))


    pre_header_array = list(max_charging_power_dict)
    header_array = ["SOC"] + pre_header_array
    SOC_array = np.arange(soc_value,81,1) #array 20 -> 80

    with open("time_real.csv","w",newline = "") as file:
        writer = csv.writer(file)
        writer.writerow(header_array)
        for i in range(len(SOC_array)):
            row = [SOC_array[i]]
            for key in max_charging_power_dict.keys():
                row.append(max_charging_power_dict[key][i])
            writer.writerow(row)
    #uppdaterad
    df2 = pd.read_csv("time_real.csv",delimiter=",")
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

    p = np.polyfit(charge_power, time_to_charge,4)

    # Generate points on the fitted curve
    charge_power_fit = np.linspace(min(charge_power), max(charge_power), 100)
    time_to_charge_fit = np.polyval(p, charge_power_fit)

    # Plot the data points and the fitted regression curve
    plt.plot(charge_power, time_to_charge, "o", label="Data")
    plt.plot(charge_power_fit, time_to_charge_fit)
    plt.legend()
    plt.xlabel("Charge Power (kW)")
    plt.ylabel("Time Passed (h)")
    plt.show()








    #plt.plot(charge_power, np.polyval(p,charge_power), "-r", label="fit")


    return("TTC (h):", np.polyval(p,charger), "End temperature (°C)",predicted_temp[0],"Energy consumed (kWh)",predicted_energy[0])

def main():

    #CSV
    df = pd.read_csv('Charging_curves.csv',delimiter=";")

    #Parametrar
    charging_powah = []
    for i in range(1,len(df.iloc[0])):
        charging_powah.append(df.iloc[0,i])
    init_soc = int(input("Enter initial state of charge: "))
    soc_value = init_soc
    final_soc = 81
    charger = int(input("Enter the charger power provided: "))
    batteritemp = 293.15

    #Spotta ut tiden
    charge_provided = charger_power_station(charger,init_soc)
    time_to_charge = time(init_soc,final_soc,charging_powah,soc_value,charger,batteritemp)
    print(time_to_charge)
    #print("the time it takes to charge is",time_to_charge,"and the charge provided is", charge_provided)

    #Ge energiförlust i kwH
    batteritemp = 293.15 #i kelvin
    #FL=energy_function(batteritemp,time_to_charge,charge_provided)
    #print(FL)



if __name__ == "__main__":
    main()
