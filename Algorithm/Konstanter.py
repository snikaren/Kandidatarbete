# -------------------------- Konstanter

m_bil = 2650                # mass car [kg]
mass_battery = 312          # mass battery [kg]
cp_battery = 900            # specific heat capacity [J/kgK] höftad från internet
cd = 0.329                  # aerodynamic drag koeff
area_front = 2.56           # Front area [m^2]
g = 9.82                    # Gravitationskonstanten [m/s^2]
t_amb = 273.15              # Ambient air temp in March [K]
c_r = 0.015                 # Rolling resistance constant #Kanske fel
my = 0.25                   # Friktionskonstant, i genomsnitt
slip = 1.05                 # Loses due to slip
den_air = 1.292             # * (273.15/t_amb) # Densitet för luften [kg/m^3]
HVCH_power = 5000           # Power for heating in cabin [W]
eta = 0.8                   # Efficiency of electric motor, from nikolce
eta_HVCH = 0.87             # Efficiency of HVAC SYS, from ahad
Battery_kWhr = 75           # Kilowatt hours for the battery, from volvo
l_battery = 2.75            # Length of batterypack [m]
w_battery = 1.5             # Width of batterypack [m]
prandtl = 0.7362            # Prandtl
visc_air = 1.338*10**(-5)   # Kinematic viscosity air [m^2/s]
k_air = 0.02364             # Thermal conductivity [W/mK]


Battery_joules = Battery_kWhr * 1000 * 3600
max_battery = 0.8 * Battery_joules
min_battery = 0.2 * Battery_joules
cells_in_series = 108



# --------------------------