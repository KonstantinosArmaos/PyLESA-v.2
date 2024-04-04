import weather
import inputs

def test_ST_weather():

    i = inputs.Inputs('commonen.xlsx', 'hp_800_ts_0')
    weather_inputs = i.weather()
    ST_weather = weather.Weather(DHI=weather_inputs['DHI'],
                                 GHI=weather_inputs['GHI'],
                                 wind_speed_10=weather_inputs['wind_speed_10'],
                                 air_temperature=weather_inputs['air_temperature'],
                                 water_temperature=weather_inputs['water_temperature']).solar_thermal()
    print (ST_weather)
test_ST_weather()

"""def test_PV_weather():

    i = inputs.weather()
    PV_weather = weather.Weather(DHI=i['DHI'], GHI=i['GHI'], DNI=i['DNI'],
                                 wind_speed_10=i['wind_speed_10'],
                                 wind_speed_50=i['wind_speed_50'],
                                 roughness_length=i['roughness_length'],
                                 pressure=i['pressure'],
                                 air_temperature=i['air_temperature'],
                                 air_density=i['air_density'],
                                 water_temperature=i['water_temperature']).PV()
    print PV_weather


def wind_weather():

    i = inputs.weather()
    wind_weather = weather.Weather(
        wind_speed_10=i['wind_speed_10'],
        wind_speed_50=i['wind_speed_50'],
        roughness_length=i['roughness_length'],
        pressure=i['pressure'],
        air_temperature=i['air_temperature']).wind_turbine()
    print wind_weather"""


# wind_weather()
