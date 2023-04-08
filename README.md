# solar-forecasting-using-LSTM
The need to forecast solar irradiation at a specific location over long-time horizons has acquired immense importance. LSTM model is used to predict solar irradiation at 10 min interval for month ahead time horizon using dataset from Killinochchi district, Faculty of Engineering, University of Jaffna Measuring Centre. 

Data was collected from Solar measuring station, Faculty of Engineering, University of Jaffna. If you need data, please contact Dr. K. Ahilan, ahilan.eng@gmail.com

The details of data logger is given below,

Wind Speed
Wind speed unit is m/s
Sensor type - Thies Anemometer Compact
sensor_height=4
formula_params=C1 var_period var_offset var_slope
var_offset=0.660000
var_slope=0.075200
var_period=1.000000

Wind direction
unit=Â°
sensor_height=4
sensor_type=wind_vane
sensor_model=Thies Wind Vane 10 Bits Serial Synchron
formula=windvane_dig
formula_params=D1 var_offset var_slope
var_offset=0.000000
var_slope=0.351562

Humidity
unit=%
sensor_height=3
sensor_type=hygro_thermo
sensor_model=Galltec Thermo-Hygro Active KP
formula=linear
formula_params=A3 var_offset var_slope
var_offset=0.000000
var_slope=100.000000

Temperature
unit=Â°C
sensor_height=3
sensor_type=hygro_thermo
sensor_model=Galltec Thermo-Hygro Active KP
formula=linear
formula_params=A2 var_offset var_slope
var_offset=-30.000000
var_slope=100.000000

Pressure
unit=mbar
sensor_label=Barometer
sensor_height=2
sensor_type=barometer
sensor_model=Barometric Pressure Sensor AB60
formula=linear
formula_params=A1 var_offset var_slope
var_offset=800.000000
var_slope=60.000000

Diffused;solar_irradiance
unit=W/m²
sensor_height=3
sensor_type=pyranometer
sensor_model=Pyranometer CMP11
formula=linear_pyr
formula_params=A5 var_sensitivity
var_sensitivity=7.890000

Global;solar_irradiance
unit=W/mÂ²

sensor_height=3
sensor_type=pyranometer
sensor_model=Pyranometer CMP11
formula=linear_pyr
formula_params=A6 var_sensitivity
var_sensitivity=7.660000

Silicon;voltage
unit=V
sensor_height=3
sensor_type=other
sensor_model=Analog Voltage
formula=linear
formula_params=A4 var_offset var_slope
var_offset=0.000000
var_slope=1.000000

Silicon irradiance sensor
Silicon Irradiance Sensor - Build as solar module - easily comparable to energy yield and system performance of PV systems.
Build as solar module - easily comparable to energy yield and system performance of PV systems

