import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pandas as pd
import csv
import random


df = pd.read_csv('Charging_curves.csv',delimiter=";")
df2 = pd.read_csv("time_series.csv",delimiter=",")

acutual_battery_capacity = 69
init_soc = 20
final_soc = 81
time = 0
#time_list = []
#soc_list = []
max_charging_power_dict = {}

#print(df.iloc[0][67])
#print(df.iloc[0][1])

#Charging (kWh) for EV
charging_powah = []

#lista med samtliga kwh
for i in range(1,len(df.iloc[0])):
    charging_powah.append(df.iloc[0,i])

#print(charging_powah)
#print(df.iloc[11,1])
#print(charging_powah)

#Stora i CSV fil: max charger power på raderna och tid,SOC som kolumner
#print(df.iloc[11,1])
a = 0 #iterrera genom charging_powah

for i in range(1,len(df.columns)):
    max_charging_power = df.iloc[11:72,i].max() #find the maximum charging power
    time_list = []   
    soc_list = []   
    init_soc = 20
    time = 0 #reset time
    battery_capacity = charging_powah[a]
    for soc in range(init_soc,final_soc):
        delta_soc = min(1,final_soc-init_soc)
        row_index = soc - 9
        charging_power = df.iloc[row_index, i]

        #Energy
        init_energy = battery_capacity * (init_soc)/100 #init_energy = battery_capacity * (init_soc/100)
        final_energy = battery_capacity * ((soc+delta_soc)/100) #final_energy = battery_capacity * ((soc+delta_soc)/100)
        energy = final_energy-init_energy #since delta_soc = 1 always this will be the battery capacity/100

        Charging_time = (energy / charging_power)/0.9 #Charging efficiency is assumed to be 90 %

        time += Charging_time

        time_list.append(time)
        #soc_list.append(init_soc)

        init_soc += delta_soc

    max_charging_power_dict[max_charging_power] = (time_list) #Dictionary with key:Max charging power -> time list for EV
    a +=1

#print(max_charging_power_dict)

#Create headers i.e the keys of the charging power dict = max_charging powers
pre_header_array = list(max_charging_power_dict)
header_array = ["SOC"] + pre_header_array

#create the CSV file (time_series)
SOC_array = np.arange(20,81,1) #array 20 -> 80
with open("time_series.csv","w",newline = "") as file:
    writer = csv.writer(file)
    writer.writerow(header_array)
    for i in range(len(SOC_array)):
        row = [SOC_array[i]]
        for key in max_charging_power_dict.keys():
            row.append(max_charging_power_dict[key][i])
        writer.writerow(row)


x_data = list(max_charging_power_dict.keys()) #samtliga charging power lista
x_data.pop(-1)
y_data = list(df2.iloc[-1,1:].values)#lista med tot tid att ladda
y_data.pop(-1)

#Modifying to get the charging for 69 kWh battery
t = 0
for value in y_data:
    y_data[t] = (acutual_battery_capacity/charging_powah[t]) * value
    t+=1


#data = list(zip(x_data, y_data))
#sorted_data = sorted(data)
#charge_power,time_to_charge = list(zip(*sorted_data))
data = list(zip(x_data, y_data))
charge_power, time_to_charge = list(zip(*data))
charge_power = list(charge_power)
time_to_charge = list(time_to_charge)


"""
#defn min och max charging power
min_power = 45
max_power = 262

#random simulaton
num_simulations = 500
random_powers = [random.uniform(min_power, max_power) for i in range(num_simulations)]

#calculate charging time 
charging_time = [] 
charging_power = [] 
for power in random_powers:
    if power>max(charge_power):
        closest_power=max(charge_power)
        time = time_to_charge[-1]
    else:
    # Find the closest charging power in the data
        for i, charge in enumerate(charge_power):
            if i == len(charge_power) - 1:
                continue
            elif power > charge and power <= charge_power[i+1]:
                closest_power = charge_power[i+1]
                time_current = time_to_charge[i]
                time = time_current + (power - closest_power) * (time_to_charge[i+1] - time_to_charge[i]) / (charge_power[i+1] - charge_power[i])

    charging_time.append(time)
    charging_power.append(power)


    #closest_power = min(charge_power, key=lambda x:abs(x-power))
    #idx = charge_power.index(closest_power)
    # Calculate the time it takes for the closest charging power
    #time_current = time_to_charge[idx]
    # linear interpolation
    #if closest_power != power:
        #if idx == len(charge_power)-1:
        
    #else:
        #time = time_current + (power - closest_power) * (time_to_charge[idx+1] - time_to_charge[idx]) / (charge_power[idx+1] - charge_power[idx])





"""
#Plot the csv file

#print(df2.columns)
#tar varje set
"""
#create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(1,len(df2.columns)):
    x = df2.iloc[:, 0]
    y = df2.iloc[:, i]
    z = float(df2.columns[i])

    # plot the data points as a scatter plot
    ax.scatter(x, y, z)


# set labels for the axes
ax.set_xlabel('SOC')
ax.set_ylabel('Time_passed (h)')
ax.set_zlabel("charging power")

# show the plot
plt.show()
"""
#PLot of time on y-axis and charging power on x-axis
"""
def exponential(x,a,b,c):
    
    return a*np.exp(-b*x)+c
"""


#Non linear regression model
p = np.polyfit(charge_power, time_to_charge,5)
x = np.linspace(0, 270, 100)
y = np.polyval(p, x)

plt.plot(charge_power, time_to_charge, "o", label="data")
plt.plot(x, y, "-", label="fit")
plt.legend()
plt.xlabel("charge power")
plt.ylabel("time passed (h)")
plt.show()



"""
with open("time_series.csv","w",newline = "") as file:
    writer = csv.writer(file)
    for i in range(len(time_list)):
        writer.writerow([time_list[i]])

#print(time_list)

# create a constant array of maximum charging power
max_power_list = [max_charging_power] * len(time_list)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(soc_list, time_list, max_power_list)
ax.set_xlabel('Initial SOC')
ax.set_ylabel('Time (hours)')
ax.set_zlabel('Charging Power (kW)')
plt.show()






for i in range(1,6):
    array_name = "array_250_" + str(i)
    locals()[array_name] = np.array([])


    array_name = "array_200_" + str(i)
    locals()[array_name] = np.array([])

    array_name = "array_150_" + str(i)
    locals()[array_name] = np.array([])

    array_name = "array_100_" + str(i)
    locals()[array_name] = np.array([])

    array_name = "array_70_" + str(i)
    locals()[array_name] = np.array([])


array_200_1 = np.array([])
array_200_2 = np.array([]) 
array_200_3 = np.array([]) 
array_200_4 = np.array([]) 
array_200_5 = np.array([]) 

for i in range(18,79):
    array_200_1= np.append(array_200_1,df.iloc[i,1])
    array_200_2= np.append(array_200_2,df.iloc[i,21]) #BMW iX50
    array_200_3= np.append(array_200_3,df.iloc[i,24]) #porche taycan
    array_200_4= np.append(array_200_4,df.iloc[i,28]) #BMW i4
    array_200_5= np.append(array_200_5,df.iloc[i,47]) #VW ID Buzz



#print(array_200_1)
SOC = np.arange(20,81,1) #array 20 -> 80


Case1 = np.array([141,134,136,136,136,125,126,128,129,117,116,118,118,113,115,110,111,106,102,100,101,99,100,95,96,90,92,89,86,87,85,80,78,75,76,74,71,71,64,65
,64,63,62,60,58,55,53,50,49,48,46,45,43,40,39,37,37,36,32,31,29
])

Case2 = np.array([141,132,135,136,136,133,133,127,128,118,121,122,116,118,113,110,109,101,104,105,102,97,100,102,97,99,97,94,91,87,85,82,83,80,78,77,70,71,72,71
,71,65,67,63,63,60,61,52,54,49,47,42,38,37,35,34,30,31,31,30,14
])

#size=SOC.size
#print(size)

X1 = SOC.reshape(-1,1)
X2 = SOC.reshape(-1,1)
X3 = SOC.reshape(-1,1)
X4 = SOC.reshape(-1,1)
X5 = SOC.reshape(-1,1)

y1 = array_200_1.reshape(-1,1)
y2 = array_200_2.reshape(-1,1)
y3 = array_200_3.reshape(-1,1)
y4 = array_200_4.reshape(-1,1)
y5 = array_200_4.reshape(-1,1)
Y = np.vstack([y1,y2,y3,y4,y5])

model = LinearRegression()
model.fit(np.vstack([X1,X2,X3,X4,X5]),Y)

slope = model.coef_[0][0]
intercept = model.intercept_[0]

equation = "y = {:.2f}x + {:.2f}".format(slope, intercept)
print(equation)


plt.scatter(np.vstack([X1,X2,X3,X4,X5]), Y)
plt.plot(np.vstack([X1,X2,X3,X4,X5]), model.predict(np.vstack([X1,X2,X3,X4,X5])), color='red')
plt.xlabel('State of Charge')
plt.ylabel('Charge Capacity')
equation = "y = {:.2f}x + {:.2f}".format(slope, intercept)
plt.annotate(equation, xy=(0.5, 0.95), xycoords='axes fraction')
plt.show()

"""
