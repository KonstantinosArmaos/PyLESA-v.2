"""generating heat demand profiles
"""
import os
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})


def house_info():

    types = ['detached',
             'semi-detached',
             'mid-terrace',
             'detached-bungalow',
             'semi-detached-bungalow',
             'ground-floor-flat',
             'mid-floor-flat',
             'top-floor-flat'
             ]

    age = ['pre-1983',
           '1983-2002',
           '2003-2007',
           'post-2007'
           ]

    dic = {'types': types, 'age': age}
    return dic


def floor_area_scaling_factor():

    # this provides a pickle full of the different possible combinatons
    # of house type and their scaling factor dependent on floor area

    types = ['detached',
             'semi-detached',
             'mid-terrace',
             'detached-bungalow',
             'semi-detached-bungalow',
             'ground-floor-flat',
             'mid-floor-flat',
             'top-floor-flat'
             ]

    age = ['pre-1983',
           '1983-2002',
           '2003-2007',
           'post-2007'
           ]

    bedrooms = [1, 2, 3, 4, 5]

    floor = {1: 70, 2: 90, 3: 110, 4: 130, 5: 150}

    floor_detached = {}
    floor_midterrace = {}
    floor_else = {}

    for x in types:
        for y in bedrooms:
            if x == 'detached':
                factor = floor[y] / 130.0
                floor_detached[y] = round(factor, 2)

            elif x == 'mid-terrace':
                factor = floor[y] / 70.0
                floor_midterrace[y] = round(factor, 2)

            else:
                factor = floor[y] / 90.0
                floor_else[y] = round(factor, 2)

    # print floor_detached, floor_midterrace, floor_else

    u = {}
    v1 = {}
    v2 = {}
    v3 = {}

    for y in age:
        v1[y] = floor_midterrace
        u['mid-terrace'] = v1

        v2[y] = floor_detached
        u['detached'] = v2

    for x in ('semi-detached',
              'detached-bungalow',
              'semi-detached-bungalow',
              'ground-floor-flat',
              'mid-floor-flat',
              'top-floor-flat'):
        for y in age:
            v3[y] = floor_else
            u[x] = v3

    df = pd.DataFrame(data=u)

    return df


def standard_profile_file():

    # this reads the excel sheet and creates a pickle
    # containing all of the standard profiles

    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "demand", 'demand.xlsx')

    df = pd.read_excel(
        path, sheet_name=None, skiprows=2, usecols="B:AG")

    for deg in ('N3', 'N2', 'N1', '0', '1', '2', '3', '4', '5', '6', '7', '8',
                '9', '10', '11', '12', '13', '14'):

        iterables = [['detached', 'semi-detached', 'mid-terrace',
                      'detached-bungalow', 'semi-detached-bungalow',
                      'ground-floor-flat', 'mid-floor-flat',
                      'top-floor-flat'],
                     ['pre-1983', '1983-2002', '2003-2007', 'post-2007']]

        index = pd.MultiIndex.from_product(iterables, names=['types', 'age'])
        df[deg].columns = index

    file = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "demand", "demand_profiles.pkl")
    with open(file, 'wb') as output_file:
        pickle.dump(
            df, output_file, protocol=pickle.HIGHEST_PROTOCOL)


def standard_day_profile(type1, age, temp):

    # gives dic of profiles at the different temps
    # for a type of building of certain age

    file = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "demand", "demand_profiles.pkl")
    df = pd.read_pickle(file)

    if temp < 0:
        temp = 'N' + str(-1 * temp)
    else:
        temp = str(temp)

    return df[temp][type1][age]


def scaled_day_profile(type1, age, bedrooms, temp):

    stp = standard_day_profile(type1, age, temp)
    scaling = floor_area_scaling_factor()[type1][age][bedrooms]

    scaled_prod = stp * scaling
    return scaled_prod


def average_day_temperature():

    path = os.path.join(
        os.path.dirname(__file__), "..", "inputs", "commonen")
    file2 = os.path.join(path, "inputs.pkl")
    resources = pd.read_pickle(file2)
    #print (resources.keys())
    print (resources['heating'].keys())
    ambient_temp = resources['resources']['air_temperature']

    df = ambient_temp.rolling(24).mean()
    df = df.round(0)
    df = df.iloc[23::24]
    df = df.reset_index(drop=True)

    return df


def profiles_from_inputs():

    path = os.path.join(
        os.path.dirname(__file__), "..", "inputs", "commonen")
    file1 = os.path.join(path, "inputs.pkl")
    heating = pd.read_pickle(file1)

    avt = average_day_temperature()

    profiles = []
    number_profiles = len(heating['heating'])
    for p in range(number_profiles):
        type1= heating['heating']['Type'][p]
        age = heating['heating']['Age'][p]
        bedrooms = heating['heating']['Bedrooms'][p]
        number_of_type = heating['heating']['Number of type'][p]
        prof = np.empty(8760)
        for day in range(365):
            temp = int(avt[day])
            if temp > 14:
                temp = 14
            elif temp < -3:
                temp = -3
            first_hour = day * 24
            last_hour = first_hour + 24
            prof[first_hour:last_hour] = (
                scaled_day_profile(type1, age, bedrooms, temp) *
                number_of_type).values
        profiles.append(prof)
    return profiles


def hot_water_addition():

    path = os.path.join(
        os.path.dirname(__file__), "..", "inputs", "commonen")
    file1 = os.path.join(path, "inputs.pkl")
    pkl_read = pd.read_pickle(file1)
    hot_water = pkl_read['hot_water']
    number_profiles = len(pkl_read['heating'])
    total_bedrooms = 0
    for p in range(number_profiles):
        total_bedrooms += pkl_read['heating']['Bedrooms'][p] * pkl_read['heating']['Number of type'][p]

    # assumed there are 1.5 people per bedroom
    total_people = total_bedrooms * 1.5

    # values in kWh / period
    hot_water_demand_per_day = total_people * hot_water
    hot_water_demand_per_hour = hot_water_demand_per_day / 24.0

    hwd = []
    for timestep in range(8760):
        hwd.append(hot_water_demand_per_hour)
    return hwd


def aggregate():

    # aggregates the profiles and dhw
    # does not account for diversity,
    # need to use smoothing algorithm in next step

    list_array = profiles_from_inputs()
    hwd = hot_water_addition()
    profile_aggregate = []
    for hour in range(8760):
        sum_demand = 0
        for x in range(len(list_array)):
            sum_demand += list_array[x][hour]
        profile_aggregate.append(sum_demand+ hwd[hour])

    return profile_aggregate


def predicted_demand():
    print ("predicted demand starting")
    def movingaverage(values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    y = aggregate()

    window = 4
    yMA = movingaverage(y, window)
    yMA_max = np.amax(yMA)
    y_max = np.amax(y)
    ratio = y_max / yMA_max
    inserting = np.array(y[: window - 1])
    inserting = ratio * inserting
    yMA = np.insert(yMA, 1, inserting)
    plt.plot(range(8760), y)
    plt.plot(range(8760), yMA)
    plt.show()
    print ("adding")
    file = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "demand", "predicted_heat_demand.csv")
    np.savetxt(file, yMA, delimiter=",", fmt='%.3e')

#def findhorn_demand():

    # validated against data from december 2017 to deceomber 2018
    # only annual consumption available

    # reduce heating by 7000kWh by shortening heating season

    # read in csv
    #file = os.path.join(
        #os.path.dirname(__file__), "..", "data",
        #"demand", "predicted_heat_demand.csv")
    #df = pd.read_csv(file, header=None, names=['dem'])
    #print df['dem'].sum()

    # # Heating from March to November only hot water
    # df['dem'][2160:7320] = 0.5025
    # # reduced by factor 0.875
    # df['dem'] = df['dem'] * 0.875
    # print df['dem'].sum()

    #plt.plot(df['dem'][0:8760], 'r', LineWidth=2)
    #plt.ylabel('Energy (kWh)')
    #plt.xlabel('Hour')
    #plt.show()

    # file = os.path.join(
    #     os.path.dirname(__file__), "..", "data",
    #     "demand", "predicted_heat_demand.csv")
    # np.savetxt(file, df['dem'].values, delimiter=",", fmt='%.3e')
