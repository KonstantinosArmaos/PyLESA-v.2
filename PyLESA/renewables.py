"""renewables module for using windpowerlib and pvlib

creates class objects for wind turbines and PV for modelling
additionally includes methods for analysis relating to modelling outputs
"""

import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import tools as t

import weather
import inputs

import pvlib
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib.modelchain import ModelChain

from windpowerlib.modelchain import ModelChain as ModelChain1
from windpowerlib.wind_turbine import WindTurbine

from windpowerlib.turbine_cluster_modelchain import TurbineClusterModelChain
from windpowerlib.wind_farm import WindFarm

# # You can use the logging package to get
# # logging messages from the windpowerlib
# # Change the logging level if you want more or less messages
# import logging
# logging.getLogger().setLevel(logging.DEBUG)
np.seterr(invalid='ignore')

def sum_renewable_generation():

    tot = wind_user_power() + wind_database_power() + PV_power() + ST_power()
    return tot


def wind_user_power(name, subname):
    """power of user-defined wind turbine

    takes inputs from the excel spreadsheet
    creates a wind turbine object
    calculates hourly power output over year

    Returns:
        pandas df -- power output 8760 steps
    """
    myInputs = inputs.Inputs(name, subname)
    input_windturbine = myInputs.windturbine_user()
    input_weather = myInputs.weather()

    myWindturbine = Windturbine(
        turbine_name=input_windturbine['turbine_name'],
        hub_height=input_windturbine['hub_height'],
        rotor_diameter=input_windturbine['rotor_diameter'],
        multiplier=input_windturbine['multiplier'],
        nominal_power=input_windturbine['nominal_power'],
        power_curve=input_windturbine['power_curve'],
        weather_input=input_weather)

    power = myWindturbine.user_power()['wind_database']
    return power


def wind_database_power(name, subname):
    """power output for a wind turbine from database

    takes inputs for weather and wind turbine from database from excel

    Returns:
        pandas df -- power output of databased wind turbine
    """

    myInputs = inputs.Inputs(name, subname)
    input_windturbine = myInputs.windturbine_database()
    input_weather = myInputs.weather()

    myWindturbine = Windturbine(
        turbine_name=input_windturbine['turbine_name'],
        hub_height=input_windturbine['hub_height'],
        rotor_diameter=input_windturbine['rotor_diameter'],
        multiplier=input_windturbine['multiplier'],
        weather_input=input_weather)

    power = myWindturbine.database_power()['wind_user']
    return power


def PV_power(name, subname):
    """power output for PV

    takes inputs from excel sheet for weather and PV

    Returns:
        pandas df -- power output for year hourly for PV
    """

    myInputs = inputs.Inputs(name, subname)
    input_PV_model = myInputs.PV_model()
    input_weather = myInputs.weather()

    myPV = PV(
        module_name=input_PV_model['module_name'],
        inverter_name=input_PV_model['inverter_name'],
        multiplier=input_PV_model['multiplier'],
        surface_tilt=input_PV_model['surface_tilt'],
        surface_azimuth=input_PV_model['surface_azimuth'],
        surface_type=input_PV_model['surface_type'],
        loc_name=input_PV_model['loc_name'],
        latitude=input_PV_model['latitude'],
        longitude=input_PV_model['longitude'],
        altitude=input_PV_model['altitude'],
        weather_input=input_weather)

    power = myPV.power_output()
    return power

def ST_power(name, subname):
    """power output for ST

    takes inputs from excel sheet for weather and ST

    Returns:
        pandas df -- power output for year hourly for ST
    """

    myInputs = inputs.Inputs(name, subname)
    input_ST_model = myInputs.ST_model()
    input_weather = myInputs.weather()
    input_demand = myInputs.demands()

    myST = Solarthermal(
        collector_type=input_ST_model['collector_type'],
        collector_area=input_ST_model['collector_area'],
        storage_capacity=input_ST_model['storage_capacity'],
        surface_tilt=input_ST_model['surface_tilt'],
        surface_azimuth=input_ST_model['surface_azimuth'],
        ground_reflectivity=input_ST_model['ground_reflectivity'],
        loc_name=input_ST_model['loc_name'],
        latitude=input_ST_model['latitude'],
        longitude=input_ST_model['longitude'],
        altitude=input_ST_model['altitude'],
        weather_input=input_weather,
        return_temp=input_demand['return_temp_DH'])

    power = myST.power_output()['ST power']
    return power

def demand_data(name, subname):
    """gets heating and dhw load demand data 

    from excel sheet

    Returns:
        df -- heating + dhw load - MJ
    """

    myInputs = inputs.Inputs(name, subname)
    input_demand_list = myInputs.demands()
    print (input_demand_list)
    input_demand = pd.DataFrame(input_demand_list, columns = ['heat_demand']) # convert to dataframe
    print (input_demand)
    input_demand.index = t.timeindex()
    monthly_demand = input_demand[['heat_demand']].resample('M').sum() * 0.27778 # from kWh to MJ
    print (monthly_demand)
    return monthly_demand

class PV(object):
    """PV class

    contains the attributes and methods for calculating PV power
    uses the PVlib library    
    """
    def __init__(self, module_name, inverter_name, multiplier,
                 surface_tilt, surface_azimuth, surface_type,
                 loc_name, latitude, longitude, altitude, weather_input):
        """initialises instance of PV class

        Arguments:
            module_name {str} -- name of PV module from database
            inverter_name {str} -- name of inverter from database
            multiplier {int} -- number of PV systems
            surface_tilt {int} -- angle from horizontal
            surface_azimuth {int} -- angle in the horizontal plane
                                    measured from south
            surface_type {str} -- surrounding surface type
            loc_name {str} -- name of location
            latitude {int} -- latitude of location
            longitude {int} --
            altitude {int} --
            weather_input {dataframe} -- dataframe with PV weather inputs
        """

        # this replaces the whitespace or invalid characters
        # with underscores so it fits with PVlib calc
        module1 = module_name.replace(' ', '_').\
            replace('-', '_').replace('.', '_').\
            replace('(', '_').replace(')', '_').\
            replace('[', '_').replace(']', '_').\
            replace(':', '_').replace('+', '_').\
            replace('/', '_').replace('"', '_').\
            replace(',', '_')
        inverter1 = inverter_name.replace(' ', '_').\
            replace('-', '_').replace('.', '_').\
            replace('(', '_').replace(')', '_').\
            replace('[', '_').replace(']', '_').\
            replace(':', '_').replace('+', '_').\
            replace('/', '_').replace('"', '_').\
            replace(',', '_')

        cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')

        if module1 in cec_modules:
            self.module = cec_modules[module1]
        elif module1 in sandia_modules:
            self.module = sandia_modules[module1]
        else:
            raise Exception('Could not retrieve PV module data')

        # of inverters and returns a pandas df
        CEC_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
        # print(CEC_inverters)
        inverter1 = 'iPower__SHO_4_8__240V_'
        self.inverter = CEC_inverters[inverter1]

        # tech parameters
        self.multiplier = multiplier
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.surface_type = surface_type

        # location parameters
        self.loc_name = loc_name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

        # weather inputs
        self.weather_input = weather_input

    def weather_data(self):
        """gets PV weather data

        gets weather data from the excel sheet for the PV

        Returns:
            df -- dhi, ghi, dni, wind_speed, temp_air
        """
        PV_weather = weather.Weather(
            DHI=self.weather_input['DHI'],
            GHI=self.weather_input['GHI'],
            DNI=self.weather_input['DNI'],
            wind_speed_10=self.weather_input['wind_speed_10'],
            air_temperature=self.weather_input['air_temperature']).PV()
        return PV_weather

    def power_output(self):
        """calculates the power output of PV

        Returns:
            df -- power output
        """

        if self.multiplier == 0:
            data = np.zeros(8760)
            df = pd.Series(data)
            return df

        location = Location(
            latitude=self.latitude, longitude=self.longitude)

        system = PVSystem(
            surface_tilt=self.surface_tilt,
            surface_azimuth=self.surface_azimuth,
            module_parameters=self.module,
            inverter_parameters=self.inverter)

        mc = ModelChain(
            system, location)
        weather = self.weather_data()

        weather.index = pd.date_range(
            start='01/01/2017', end='01/01/2018',
            freq='1h', tz='Europe/London', inclusive = 'left')
        # times = naive_times.tz_localize('Europe/London')
        # print weather
        # weather = pd.DataFrame(
        #     [[1050, 1000, 100, 30, 5]],
        #     columns=['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed'],
        #     index=[pd.Timestamp('20170401 1200', tz='Europe/London')])

        mc.run_model(weather=weather)
        # multiply by system losses
        # also multiply by correction factor
        power = (mc.ac.fillna(value=0).round(2) *
                 self.multiplier * 0.001 * 0.85 * 0.85)
        # remove negative values
        power[power < 0] = 0
        # multiply by monthly correction factors
        # for correcting MEERA solar data set
        cf = [1.042785944,
              1.059859907,
              1.037299072,
              0.984286745,
              0.995849527,
              0.973795815,
              1.003315908,
              1.014427134,
              1.046833,
              1.091837017,
              1.039504694,
              0.95520793]

        # multiply by correction factor for each hour
        for hour in range(8760):
            month = int(math.floor(hour / 730))
            power[hour] = power[hour] * cf[month]

        power = power.reset_index(drop=True)

        return power

class Solarthermal(object):
    """Solar thermal class

    contains the attributes and methods for calculating solar thermal power
    
    uses f-chart method
    """

    def __init__(self, collector_type, collector_area, storage_capacity,
                 surface_tilt, surface_azimuth, ground_reflectivity,
                 loc_name, latitude, longitude, altitude, weather_input, return_temp):
        """initialises instance of ST class

        Arguments:
            collector_type {str} -- type of solar thermal collector from user input
            collector_area {str} -- total area of solar thermal collectors
            storage_capacity {int} -- total hot water storage capacity (l)
            surface_tilt {int} -- angle from horizontal
            surface_azimuth {int} -- angle in the horizontal plane
                                    measured from south
            ground_reflectivity {float} -- 
            loc_name {str} -- name of location
            latitude {int} -- latitude of location
            longitude {int} --
            altitude {int} --
            weather_input {dataframe} -- dataframe with PV weather inputs
        """

        # tech parameters
        self.collector_type = collector_type
        self.collector_area = collector_area
        self.storage_capacity = storage_capacity
        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth
        self.ground_reflectivity = ground_reflectivity

        # location parameters
        self.loc_name = loc_name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

        # weather inputs
        self.weather_input = weather_input

        self.return_temp = return_temp

        """self.folder_path = os.path.join(
            os.path.dirname(__file__), "..", "inputs", name[:-5], subname + '.pkl')"""
        
         
    def weather_data(self):
        """gets solar thermal weather data

        gets weather data from the excel sheet for the solar thermal

        Returns:
            df -- dhi, ghi, temp_air
        """
    
        ST_weather = weather.Weather(
            DHI=self.weather_input['DHI'],
            GHI=self.weather_input['GHI'],
            air_temperature=self.weather_input['air_temperature'],
            water_temperature=self.weather_input['water_temperature']).solar_thermal()
        #print (ST_weather)
        """monthly_sum = ST_weather[['dhi', 'ghi']].resample('M').sum()  
        monthly_average = ST_weather[['air_temperature', 'water_temperature']].resample('M').mean()

        ST_weather_combined = pd.concat([monthly_sum, monthly_average], axis = 1)
        print (ST_weather_combined)"""
        return ST_weather
    

    def power_output(self):
        """calculates the power output of ST

        Returns:
            df -- power output
        """
        weather = self.weather_data()

        # constants below from Bellos et al 2022 paper 
        # https://doi.org/10.3390/app12094566

        if self.collector_type == 'Simple Flat Plate':
            collector_constants = {
                'ta_ratio': 0.91,
                'fr_ratio': 0.95,
                'fr_ta': 0.73,
                'fr_ul': 5.85
            }
        if self.collector_type == 'Advanced Flat Plate':
            collector_constants = {
                'ta_ratio': 0.91,
                'fr_ratio': 0.95,
                'fr_ta': 0.77,
                'fr_ul': 4.59
            }
        if self.collector_type == 'Evacuated Tube':
            collector_constants = {
                'ta_ratio': 0.91,
                'fr_ratio': 0.95,
                'fr_ta': 0.70,
                'fr_ul': 2.92
            }


        if self.collector_area == 0:
            data = np.zeros(8760)
            df = pd.DataFrame(data, columns = ['ST power'])
            return df

        ST_power = np.array([])
        R_b_array = np.array([])

        for day in range(1,366):
            delta = np.radians(23.45) * np.sin(2 * np.pi * (day + 284) / 365)
            omega_s = min([np.arccos( -np.tan(np.radians(self.latitude)) * np.tan(delta)), np.arccos( -np.tan(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.tan(delta))])
            R_b = (np.cos(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.cos(delta) * omega_s + np.pi / 180 * omega_s * np.sin(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.sin(delta)) / (np.cos(np.radians(self.latitude)) * np.cos(delta) * np.sin(omega_s) + np.pi / 180 * omega_s * np.sin(np.radians(self.latitude)) * np.sin(delta))
            R_b_array = np.append(R_b_array,R_b)
            for hour in range(24):
                hour_of_year = hour + (day - 1) * 24
                critical_radiation = (collector_constants['fr_ul'] * (self.return_temp - weather['air_temperature'].iloc[hour_of_year])) / collector_constants['fr_ta']
                incident_radiation = ((weather['ghi'].iloc[hour_of_year] - weather['dhi'].iloc[hour_of_year]) * R_b_array[day-1] + 0.5 * weather['dhi'].iloc[hour_of_year] * (1 + np.cos(self.surface_tilt)) + 0.5 * self.ground_reflectivity * weather['ghi'].iloc[hour_of_year] * (1 - np.cos(self.surface_tilt)))
                utilisable_energy = self.collector_area * collector_constants['fr_ta'] * (incident_radiation - critical_radiation) / 1000 #in kW
                ST_power = np.append(ST_power, utilisable_energy)
                ST_power = np.maximum(ST_power,0)
        df = pd.DataFrame(ST_power)
        df.columns = ['ST power']
        return df

        """for key in mean_day:
            delta = np.radians(23.45) * np.sin(2 * np.pi * (mean_day[key]+284)/365)
            omega_s = min([np.arccos( -np.tan(np.radians(self.latitude)) * np.tan(delta)), np.arccos( -np.tan(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.tan(delta))])
            R_b = (np.cos(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.cos(delta) * omega_s + np.pi / 180 * omega_s * np.sin(np.radians(self.latitude)-np.radians(self.surface_tilt)) * np.sin(delta)) / (np.cos(np.radians(self.latitude)) * np.cos(delta) * np.sin(omega_s) + np.pi / 180 * omega_s * np.sin(np.radians(self.latitude)) * np.sin(delta))
            # x0.27778/30 to convert from kWh to MJ and from monthly to daily 
            H_t = 1/1000*(((weather['ghi'].iloc[key-1] - weather['dhi'].iloc[key-1]) * R_b + 0.5 * weather['dhi'].iloc[key-1] * (1 + np.cos(self.surface_tilt)) + 0.5 * self.ground_reflectivity * weather['ghi'].iloc[key-1] * (1 - np.cos(self.surface_tilt))) / (30 * 0.27778))
            dhw_correction = (11.6 + 1.18 * 55 + 3.86 * weather['water_temperature'].iloc[key-1] - 2.32 * weather['air_temperature'].iloc[key-1]) / (100 - weather['air_temperature'].iloc[key-1])
            x_fchart = dhw_correction * collector_constants['fr_ul'] * collector_constants['fr_ratio'] * (100 - weather['air_temperature'].iloc[key-1]) * 2592000 * self.collector_area / (monthly_load[key]/0.27778) / 10**6
            y_fchart = collector_constants['fr_ta'] * collector_constants['ta_ratio'] * collector_constants['fr_ratio'] * H_t * days_month[key] * self.collector_area / (monthly_load[key]/0.27778)
            solar_fraction = 1.029 * y_fchart - 0.065 * x_fchart - 0.245 * y_fchart ** 2 + 0.0018 * x_fchart ** 2 + 0.02 * y_fchart ** 3

            delta_list.append(delta)
            omega_s_list.append(omega_s)
            R_b_list.append(R_b)
            H_t_list.append(H_t)
            x_fchart_list.append(x_fchart)
            y_fchart_list.append(y_fchart)
            f_list.append(solar_fraction)
        
        print ('delta', delta_list)
        print ('omega', omega_s_list)
        print ('rb', R_b_list)
        print ('ht', H_t_list)
        print ('X: ', x_fchart_list)
        print ('Y: ', y_fchart_list)
        print ('solar fractions: ', f_list)

        return f_list"""

class Windturbine(object):

    def __init__(self, turbine_name, hub_height,
                 rotor_diameter, multiplier, weather_input,
                 nominal_power=None, power_curve=None,
                 wind_farm_efficiency=None):
        """
        wind turbine class

        class for modelling user and database defined equations

        Arguments:
            turbine_name {str} -- name of turbine, only matters for database
            hub_height {float} -- height of turbine
            rotor_diameter {float} --
            multiplier {int} -- number of wind turbines of type
            weather_input {pandas df} -- use weather class wind instance method

        Keyword Arguments (for user-defined turbine):
            nominal_power {float} -- (default: {None})
            power_curve {[dict ]} -- dict with power curve
                                     values (default: {None})
        """

        self.turbine_name = turbine_name
        self.hub_height = hub_height
        self.rotor_diameter = rotor_diameter
        self.multiplier = multiplier
        self.weather_input = weather_input

        self.nominal_power = nominal_power
        self.power_curve = power_curve
        self.wind_farm_efficiency = wind_farm_efficiency

    def weather_data(self):
        """input weather data

        creates weather object with necessary attributes

        Returns:
            pandas df -- hourly weather data for wind modelling
        """

        wind_weather = weather.Weather(
            wind_speed_10=self.weather_input['wind_speed_10'],
            wind_speed_50=self.weather_input['wind_speed_50'],
            roughness_length=self.weather_input['roughness_length'],
            pressure=self.weather_input['pressure'],
            air_temperature=self.weather_input['air_temperature']).wind_turbine()
        return wind_weather

    def user_power(self):
        """wind power output for user-defined turbine

        inputs weather and turbine spec
        initialises the wind turbine object
        runs model
        calculates power output

        Returns:
            pandas df -- hourly year power output
        """

        if self.multiplier == 0:
            data = np.zeros(8760)
            df = pd.DataFrame(data, columns=['wind_user'])
            return df

        multi = self.multiplier

        # this returns dict which contains all the info for the windturbine
        myTurbine = {'name': self.turbine_name,
                     'nominal_power': self.nominal_power,
                     'hub_height': self.hub_height,
                     'rotor_diameter': self.rotor_diameter,
                     'power_curve': self.power_curve
                     }

        # input weather data
        weather = self.weather_data()

        # initialises a wind turbine object using
        # the WindTurbine class from windpowerlib
        UserTurbine = WindTurbine(**myTurbine)
        mc_my_turbine = ModelChain1(UserTurbine).run_model(weather)

        # 1000 factor to make it into kW,
        # and the multi to give number of wind turbines
        # multiply by 0.5 to correct MEERA dataset if needed
        series = (mc_my_turbine.power_output * multi / 1000.).round(2)
        df = series.to_frame()
        df.columns = ['wind_user']
        df = df.reset_index(drop=True)

        return df

    def database_power(self):
        """wind turbine database power output

        power output calculation
        initialise ModelChain with default
        parameters and use run_model method
        to calculate power output

        Returns:
            pandas df -- hourly year of power output
        """

        if self.multiplier == 0:
            data = np.zeros(8760)
            df = pd.DataFrame(data, columns=['wind_database'])
            return df

        multi = self.multiplier

        # specification of wind turbine where
        # power coefficient curve and nominal
        # power is provided in an own csv file

        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", 'oedb')#, 'power_curves.csv')

        myTurbine = {
            'turbine_type': self.turbine_name,  # turbine type as in file
            'hub_height': self.hub_height,  # in m
            'rotor_diameter': self.rotor_diameter,  # in m
            'path': csv_path  # using power_curve csv
        }

        # own specifications for ModelChain setup
        modelchain_data = {
            'wind_speed_model': 'logarithmic',  # 'logarithmic' (default),
                                                # 'hellman' or
                                                # 'interpolation_extrapolation'
            'density_model': 'barometric',  # 'barometric' (default), 'ideal_gas' or
                                           # 'interpolation_extrapolation'
            'temperature_model': 'linear_gradient',  # 'linear_gradient' (def.) or
                                                     # 'interpolation_extrapolation'
            'power_output_model': 'power_curve',  # 'power_curve' (default) or
                                                  # 'power_coefficient_curve'
            'density_correction': False,  # False (default) or True
            'obstacle_height': 0,  # default: 0
            'hellman_exp': None}  # None (default) or None

        # initialise WindTurbine object
        turbineObj = WindTurbine(**myTurbine)

        weather = self.weather_data()

        mc_my_turbine = ModelChain1(turbineObj, **modelchain_data).run_model(weather)
        # write power output timeseries to WindTurbine object
        # divide by 1000 to keep in kW
        # multply by 0.5 for correcting reanalysis dataset
        series = (mc_my_turbine.power_output * multi * 0.5 / 1000).round(2)
        df = series.to_frame()
        df.columns = ['wind_database']
        df = df.reset_index(drop=True)

        return df

    def wind_farm_power(self):

        # specification of wind turbine where
        # power coefficient curve and nominal
        # power is provided in an own csv file

        if self.multiplier == 0:
            data = np.zeros(8760)
            df = pd.DataFrame(data, columns=['wind_farm'])
            return df

        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", 'oedb')
        myTurbine = {
            'turbine_type': self.turbine_name,  # turbine type as in file
            'hub_height': self.hub_height,  # in m
            'rotor_diameter': self.rotor_diameter,  # in m
            'path': csv_path  # using power_curve csv
        }

        # initialise WindTurbine object
        siemens = WindTurbine(**myTurbine)

        # specification of wind farm data
        farm = {
            'name': 'example_farm',
            'wind_turbine_fleet': [
                {'wind_turbine': siemens,
                 'number_of_turbines': self.multiplier}],
            'efficiency': self.wind_farm_efficiency}

        # initialize WindFarm object
        farm_obj = WindFarm(**farm)

        weather = self.weather_data()

        # power output calculation for example_farm
        # initialize TurbineClusterModelChain with default parameters and use
        # run_model method to calculate power output
        mc_farm = TurbineClusterModelChain(farm_obj).run_model(weather)
        # write power output time series to WindFarm object
        farm_obj.power_output = mc_farm.power_output

        # units in kWh
        # times by 0.5 to correct for MEERA dataset
        series = (farm_obj.power_output * 0.5 / 1000).round(2)
        df = series.to_frame()
        df.columns = ['wind_farm']

        return df
