import os
import pandas as pd
import numpy as np
import shutil
import pickle
# import register_matplotlib_converters
import matplotlib.pyplot as plt

# 3d plotting
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import inputs
import tools as t
import grid

"""
plt.style.available

['seaborn-bright',
'seaborn-darkgrid',
'seaborn-white',
'ggplot',
'seaborn-dark',
'seaborn-muted',
'seaborn-notebook',
'seaborn-paper',
'classic',
'grayscale',
'seaborn-whitegrid',
'fivethirtyeight',
'bmh',
'seaborn-ticks',
'seaborn-dark-palette',
'seaborn-pastel',
'dark_background',
'seaborn-deep',
'seaborn-colorblind',
'seaborn-talk',
'seaborn-poster']
"""
# register_matplotlib_converters()
# plt.style.use('seaborn-paper')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 6})


def run_plots(name, subname):

    myInputs = inputs.Inputs(name, subname)
    # controller inputs
    controller_info = myInputs.controller()['controller_info']
    timesteps = controller_info['total_timesteps']

    if timesteps == 8760:
        for period in ['Year', 'Winter', 'Summer']:
            myPlots = Plot(name, subname, period)
            myPlots.operation()
            myPlots.elec_demand_and_RES()
            myPlots.HP_and_heat_demand()
            myPlots.TS()
            #myPlots.ES()
            myPlots.grid()
            if period == 'Year':
                myPlots.RES_bar()
                myPlots.ST_bar()
                myPlots.CO2_bar()
                myPlots.import_export_bar()
    else:
        period = 'User'
        myPlots = Plot(name, subname, period)
        myPlots.operation()
        myPlots.elec_demand_and_RES()
        myPlots.HP_and_heat_demand()
        myPlots.TS()
        #myPlots.ES()
        myPlots.grid()
        myPlots.RES_bar()


def run_KPIs(name):
    
    my3DPlots = ThreeDPlots(name)
    my3DPlots.KPIs_to_csv()
    #my3DPlots.plot_opex()
    #my3DPlots.plot_RES()
    #my3DPlots.plot_heat_from_RES()
    #my3DPlots.plot_HP_size_ratio()
    #my3DPlots.plot_HP_utilisation()
    #my3DPlots.plot_capital_cost_hp()
    #my3DPlots.plot_LCOH()
    #my3DPlots.plot_carbon_emissions()
    #my3DPlots.plot_COH()

class Plot(object):

    def __init__(self, name, subname, period):

        self.folder_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs",
            name[:-5], subname, period)
        self.name = name
        self.subname = subname
        # period can be 'Summer', 'Winter', 'Year', 'User'
        self.period = period

        self.file_output_pickle = os.path.join(
            os.path.dirname(__file__), '..', 'outputs',
            name[:-5], subname, 'outputs.pkl')
        self.results = pd.read_pickle(self.file_output_pickle)

        self.myInputs = inputs.Inputs(name, subname)
        # controller inputs
        controller_info = self.myInputs.controller()['controller_info']
        self.timesteps = controller_info['total_timesteps']
        self.first_hour = controller_info['first_hour']

        # creates a folder for keeping all the
        # outputs as saved from the excel file
        if os.path.isdir(self.folder_path) is False:
            os.mkdir(self.folder_path)

        elif os.path.isdir(self.folder_path) is True:
            shutil.rmtree(self.folder_path)
            os.mkdir(self.folder_path)

    def period_timesteps(self):

        # if it is a full year simulated
        # plot the period from class attribute
        if self.timesteps == 8760:

            if self.period == 'Summer':
                timesteps = 168
                first_hour = 4380
            elif self.period == 'Winter':
                timesteps = 168
                first_hour = 100
            elif self.period == 'Year':
                timesteps = 8760
                first_hour = 0
            elif self.period == 'User':
                timesteps = self.timesteps
                first_hour = self.first_hour
            else:
                raise Exception('Unacceptable period defined')

        else:
            timesteps = self.timesteps
            first_hour = 0

        return {'first_hour': first_hour, 'timesteps': timesteps}

    def operation(self):

        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'operation.png')
        # pp = PdfPages(fileout)

        results = self.results
        timesteps = self.timesteps

        ST = []
        HPt = []
        aux = []
        final_nodes_temp = []
        IC = []
        surplus = []
        hd = []
        export = []
        for i in range(timesteps):
            ST.append(results[i]['ST']['heat_total_output'])
            HPt.append(results[i]['HP']['heat_total_output'])
            hd.append(results[i]['heat_demand']['heat_demand'])
            aux.append(results[i]['aux']['demand'])
            final_nodes_temp.append(results[i]['TS']['final_nodes_temp'])
            IC.append(results[i]['grid']['import_price'])
            surplus.append(results[i]['grid']['surplus'])
            export.append(results[i]['grid']['total_export'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()

        plt.subplot(4, 1, 1)
        plt.title('Operation graphs')
        plt.plot(time, ST[first_hour:final_hour], 'magenta', linewidth=1)
        plt.plot(time, HPt[first_hour:final_hour], 'r', linewidth=1)
        plt.plot(time, aux[first_hour:final_hour], 'b', linewidth=1)
        plt.plot(time, hd[first_hour:final_hour], 'g', linewidth=1)
        plt.xlabel('Hour of the year')
        plt.ylabel('Energy (kWh)')
        plt.legend(['ST', 'HPt', 'aux', 'HD'], loc='best')

        plt.subplot(4, 1, 2)
        plt.plot(time, final_nodes_temp[first_hour:final_hour],
                 'b', linewidth=1)
        plt.ylabel('Node temperature \n (degC)')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 3)
        plt.plot(time, IC[first_hour:final_hour], 'g', linewidth=1)
        plt.ylabel('Import cost \n (Pounds per MWh)')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 4)
        plt.plot(time, surplus[first_hour:final_hour], 'm', linewidth=1)
        plt.plot(time, export[first_hour:final_hour], 'b', linewidth=1)
        plt.legend(['surplus', 'export'], loc='best')
        plt.ylabel('Energy (kWh)')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def elec_demand_and_RES(self):

        # points to where the pdf will be saved and its name
        fileout = os.path.join(
            self.folder_path,
            'electricity_demand_and_generation.png')

        results = self.results
        timesteps = self.timesteps
        first_hour = self.first_hour
        final_hour = first_hour + timesteps

        RES = []
        ES = []
        imp = []
        dem = []
        generation_total = []
        wind = []
        PV = []
        elec_demand = []
        HP = []
        aux = []
        export = []

        for i in range(timesteps):
            RES.append(results[i]['elec_demand']['RES'])
            ES.append(results[i]['elec_demand']['ES'])
            imp.append(results[i]['elec_demand']['import'])
            dem.append(results[i]['elec_demand']['elec_demand'])
            generation_total.append(results[i]['RES']['generation_total'])
            wind.append(results[i]['RES']['wind'])
            PV.append(results[i]['RES']['PV'])
            elec_demand.append(results[i]['RES']['elec_demand'])
            HP.append(results[i]['RES']['HP'])
            aux.append(results[i]['RES']['aux'])
            export.append(results[i]['RES']['export'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()

        plt.subplot(3, 1, 1)
        plt.stackplot(
            time,
            RES[first_hour:final_hour],
            imp[first_hour:final_hour],
            ES[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.xlabel('Hour of the year')
        plt.legend(['RES', 'Import', 'ES'], loc='best')
        plt.title('Electrical demand')

        # Plot stack of RES generation
        plt.subplot(3, 1, 2)
        plt.stackplot(
            time,
            wind[first_hour:final_hour],
            PV[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.xlabel('Hour of the year')
        plt.legend(['Wind', 'PV'], loc='best')
        plt.title('Renewable power generation')

        # Plot stack of RES usage
        plt.subplot(3, 1, 3)
        plt.stackplot(
            time,
            elec_demand[first_hour:final_hour],
            HP[first_hour:final_hour],
            aux[first_hour:final_hour],
            export[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.legend(['Elec demand', 'HP', 'Aux', 'Export'], loc='best')
        plt.title('Renewable power usage')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def HP_and_heat_demand(self):

        # points to where the pdf will be saved and its name
        fileout = os.path.join(
            self.folder_path, 'heat_demand_and_heat_pump.png')
        # pp = PdfPages(fileout)

        results = self.results
        timesteps = self.timesteps
        first_hour = self.first_hour
        final_hour = first_hour + timesteps

        HP = []
        TS = []
        ST = []
        aux = []
        hd = []

        HP_heat_to_heat_demand = []
        HP_heat_to_TS = []
        HP_heat_total_output = []
        
        ST_heat_to_heat_demand = []
        ST_heat_to_TS = []
        ST_heat_total_output = []

        elec_total_usage = []
        elec_RES_usage = []
        elec_ES_usage = []
        elec_import_usage = []

        cop = []
        duty = []

        for i in range(timesteps):
            HP.append(results[i]['heat_demand']['HP'])
            TS.append(results[i]['heat_demand']['TS'])
            ST.append(results[i]['ST']['heat_to_heat_demand'])
            aux.append(
                results[i]['heat_demand']['heat_demand'] -
                results[i]['heat_demand']['HP'] -
                results[i]['heat_demand']['TS'] - 
                results[i]['ST']['heat_to_heat_demand'])
            hd.append(results[i]['heat_demand']['heat_demand'])
            HP_heat_to_heat_demand.append(
                results[i]['HP']['heat_to_heat_demand'])
            HP_heat_to_TS.append(
                results[i]['HP']['heat_to_TS'])
            HP_heat_total_output.append(
                results[i]['HP']['heat_total_output'])
            ST_heat_to_heat_demand.append(
                results[i]['ST']['heat_to_heat_demand'])
            ST_heat_to_TS.append(
                results[i]['ST']['heat_to_TS'])
            ST_heat_total_output.append(
                results[i]['ST']['heat_total_output'])
            elec_total_usage.append(
                results[i]['HP']['elec_total_usage'])
            elec_RES_usage.append(
                results[i]['HP']['elec_RES_usage'])
            elec_ES_usage.append(
                results[i]['HP']['elec_from_ES_to_demand'])
            elec_import_usage.append(
                results[i]['HP']['elec_import_usage'])
            cop.append(
                results[i]['HP']['cop'])
            duty.append(
                results[i]['HP']['duty'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        av_cop = round(sum(cop)/len(cop),1)

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()

        plt.subplot(4, 1, 1)
        plt.stackplot(
            time,
            ST[first_hour:final_hour],
            HP[first_hour:final_hour],
            TS[first_hour:final_hour],
            aux[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.legend(['ST', 'HP', 'TS', 'Aux'], loc='best')
        plt.title('Heat demand')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 2)
        plt.stackplot(
            time,
            HP_heat_to_heat_demand[first_hour:final_hour],
            HP_heat_to_TS[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.legend(['Heat demand', 'TS'], loc='best')
        plt.title('Heat pump thermal output')
        plt.xlabel('Hour of the year')

        # Plot stack of HP electricity usage
        plt.subplot(4, 1, 3)
        plt.stackplot(
            time,
            elec_RES_usage[first_hour:final_hour],
            elec_import_usage[first_hour:final_hour],
            elec_ES_usage[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.legend(['RES usage', 'Import', 'ES'], loc='best')
        plt.title('Heat pump electrical usage')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 4)
        plt.stackplot(
            time,
            ST_heat_to_heat_demand[first_hour:final_hour],
            ST_heat_to_TS[first_hour:final_hour])
        plt.ylabel('Energy (kWh)')
        plt.legend(['Heat demand', 'TS'], loc='best')
        plt.title('Solar thermal thermal output')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(time, cop[first_hour:final_hour], 'r', linewidth=1)
        plt.ylabel('COP')
        #plt.legend(['Average COP ' + str(av_cop)], loc='best')
        plt.title('Heat pump performance')
        plt.xlabel('Hour of the year')

        plt.subplot(2, 1, 2)
        plt.plot(time, duty[first_hour:final_hour], 'g', linewidth=1)
        plt.ylabel('Duty (kW)')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        fileout = os.path.join(
            self.folder_path, 'heatpump_cop_and_duty.png')
        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def TS(self):

        # points to where the pdf will be saved and its name
        fileout = os.path.join(
            self.folder_path,
            'thermal_storage.png')

        results = self.results
        timesteps = self.timesteps
        first_hour = self.first_hour
        final_hour = first_hour + timesteps

        charging_total = []
        discharging_total = []
        final_nodes_temp = []

        for i in range(timesteps):
            charging_total.append(results[i]['TS']['charging_total'])
            discharging_total.append(results[i]['TS']['discharging_total'])
            final_nodes_temp.append(results[i]['TS']['final_nodes_temp'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(
            time,
            charging_total[first_hour:final_hour],
            'r', linewidth=1)
        plt.ylabel('Energy (kWh)')
        plt.legend(['Charging'], loc='best')
        plt.title('Thermal storage')
        plt.xlabel('Hour of the year')

        # Plot stack of RES generation
        plt.subplot(3, 1, 2)
        plt.plot(
            time,
            discharging_total[first_hour:final_hour],
            'b', linewidth=1)
        plt.ylabel('Energy (kWh)')
        plt.legend(['Discharging'], loc='best')
        plt.xlabel('Hour of the year')

        # Plot stack of RES usage
        plt.subplot(3, 1, 3)
        plt.plot(
            time, final_nodes_temp[first_hour:final_hour],
            linewidth=1)
        plt.ylabel('Temperature degC')
        leg = []
        for x in range(len(final_nodes_temp[first_hour])):
            leg.append(str(x + 1))
        plt.legend(leg, loc='best')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def ES(self):
        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'electrical_storage.png')

        results = self.results
        timesteps = self.timesteps
        first_hour = self.first_hour
        final_hour = first_hour + timesteps

        ES_to_demand = []
        ES_to_HP_to_demand = []
        RES_to_ES = []
        import_for_ES = []
        soc = []
        IC = []
        surplus = []
        export = []
        # dem = []

        for i in range(timesteps):
            ES_to_demand.append(-1 * results[i]['ES']['discharging_to_demand'])
            ES_to_HP_to_demand.append(-1 * results[i]['ES']['discharging_to_HP'])
            RES_to_ES.append(results[i]['ES']['charging_from_RES'])
            import_for_ES.append(results[i]['ES']['charging_from_import'])
            soc.append(results[i]['ES']['final_soc'])
            IC.append(results[i]['grid']['import_price'])
            surplus.append(results[i]['grid']['surplus'])
            export.append(results[i]['grid']['total_export'])
            # dem.append(results[i]['elec_demand']['elec_demand'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title('Electrical Storage')
        plt.plot(time, ES_to_demand[first_hour:final_hour], 'r', linewidth=1)
        plt.plot(time, ES_to_HP_to_demand[first_hour:final_hour], 'y', linewidth=1)
        plt.plot(time, RES_to_ES[first_hour:final_hour], 'b', linewidth=1)
        plt.plot(time, import_for_ES[first_hour:final_hour], 'g', linewidth=1)
        plt.ylabel('ES c/d')
        plt.legend(['ES_to_demand', 'ES_to_HP_to_demand', 'RES_to_ES', 'import_for_ES'], loc='best')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 2)
        plt.plot(time, soc[first_hour:final_hour], 'r', linewidth=1)
        # plt.plot(time, dem[first_hour:final_hour], 'g', LineWidth=1)
        plt.ylabel('SOC')
        plt.legend(['SOC'], loc='best')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 3)
        plt.plot(time, IC[first_hour:final_hour], 'g', linewidth=1)
        plt.ylabel('Import cost')
        plt.xlabel('Hour of the year')

        plt.subplot(4, 1, 4)
        plt.plot(time, surplus[first_hour:final_hour], 'm', linewidth=1)
        plt.plot(time, export[first_hour:final_hour], 'b', linewidth=1)
        plt.legend(['surplus', 'export'], loc='best')
        plt.ylabel('Surplus, and export')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def grid(self):

        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'grid.png')

        results = self.results
        timesteps = self.timesteps
        first_hour = self.first_hour
        final_hour = first_hour + timesteps

        total_export = []
        total_import = []
        import_price = []
        cashflow = []

        for i in range(timesteps):
            total_export.append(results[i]['grid']['total_export'])
            total_import.append(results[i]['grid']['total_import'])
            import_price.append(results[i]['grid']['import_price'])
            cashflow.append(results[i]['grid']['cashflow'])

        pt = self.period_timesteps()
        first_hour = pt['first_hour']
        timesteps = pt['timesteps']
        final_hour = first_hour + timesteps

        # Plot solution
        time = range(first_hour, final_hour)
        plt.figure()

        plt.subplot(3, 1, 1)
        plt.title('Grid interaction')
        plt.plot(time, total_import[first_hour:final_hour], 'b', linewidth=1)
        plt.plot(time, total_export[first_hour:final_hour], 'r', linewidth=1)
        plt.ylabel('Energy (kWh)')
        plt.legend(['Import', 'Export'], loc='best')
        plt.xlabel('Hour of the year')

        plt.subplot(3, 1, 2)
        plt.plot(time, import_price[first_hour:final_hour],
                 'g', linewidth=1)
        plt.ylabel('import price \n (Pounds/MWh)')
        plt.xlabel('Hour of the year')

        plt.subplot(3, 1, 3)
        plt.plot(time, cashflow[first_hour:final_hour], 'y', linewidth=1)
        plt.ylabel('Cashflow \n (Pounds/MWh)')

        plt.xlabel('Hour of the year')
        plt.tight_layout()

        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def RES_bar(self):

        # bar chart of wind
        # bar chart of pv
        # bar chart of total RES

        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'RES_bar_charts.png')

        results = self.results
        timesteps = self.timesteps

        PV = np.zeros(timesteps)
        #wind = np.zeros(timesteps)
        ST = np.zeros(timesteps)
        RES = np.zeros(timesteps)

        for i in range(timesteps):
            RES[i] = results[i]['RES']['generation_total']
            #wind[i] = results[i]['RES']['wind']
            PV[i] = results[i]['RES']['PV']
            ST[i] = results[i]['ST']['heat_total_output']

        #wind_monthly = t.sum_monthly(wind)
        #wind_year = round(wind.sum() / 1000, 2)

        PV_monthly = t.sum_monthly(PV)
        PV_year = round(PV.sum() / 1000, 2)

        ST_monthly = t.sum_monthly(ST)
        ST_year = round(ST.sum() / 1000, 2)

        RES_monthly = t.sum_monthly(RES)
        RES_year = round(RES.sum() / 1000, 2)

        # bar chart of months
        plt.figure()

        """plt.subplot(3, 1, 1)
        plt.bar(
            range(len(wind_monthly)),
            list(wind_monthly.values()),
            align='center')
        plt.xticks(range(len(wind_monthly)), list(wind_monthly.keys()))
        plt.yticks()
        plt.ylabel('Energy (MWh)')
        plt.title('Total Wind Production (MWh): %s' % (wind_year))"""

        plt.subplot(3, 1, 1)
        plt.bar(
            range(len(PV_monthly)),
            list(PV_monthly.values()),
            align='center')
        plt.xticks(range(len(PV_monthly)), list(PV_monthly.keys()))
        plt.yticks()
        plt.ylabel('Energy (MWh)')
        plt.title('Total PV Production (MWh): %s' % (PV_year))

        plt.subplot(3, 1, 2)
        plt.bar(
            range(len(ST_monthly)),
            list(ST_monthly.values()),
            align='center')
        plt.xticks(range(len(ST_monthly)), list(ST_monthly.keys()))
        plt.yticks()
        plt.ylabel('Energy (MWh)')
        plt.title('Total ST Production (MWh): %s' % (ST_year))

        plt.subplot(3, 1, 3)
        plt.bar(
            range(len(RES_monthly)),
            list(RES_monthly.values()),
            align='center')
        plt.xticks(range(len(RES_monthly)), list(RES_monthly.keys()))
        plt.yticks()
        plt.ylabel('Energy (MWh)')
        plt.title('Total RES Production (MWh): %s' % (RES_year))

        plt.tight_layout()
        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def ST_bar(self):

        # monthly bar chart of ST solar fraction

        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'ST_bar_chart.png')

        results = self.results
        timesteps = self.timesteps

        ST_hd = np.zeros(timesteps)
        ST_ts = np.zeros(timesteps)
        hd_tot = np.zeros(timesteps)
        ST_tot = np.zeros(timesteps)

        for i in range(timesteps):
            ST_hd[i] = results[i]['ST']['heat_to_heat_demand']
            ST_ts[i] = results[i]['ST']['heat_to_TS']
            hd_tot[i] = results[i]['heat_demand']['heat_demand']
            ST_tot[i] = results[i]['ST']['heat_total_output']


        hd_tot_monthly = t.sum_monthly(hd_tot)
        hd_tot_year = round(hd_tot.sum() / 1000, 2)


        ST_hd_monthly = t.sum_monthly(ST_hd)
        ST_hd_year = round(ST_hd.sum() / 1000, 2)

        ST_ts_monthly = t.sum_monthly(ST_ts)
        ST_ts_year = round(ST_ts.sum() / 1000, 2)

        ST_tot_monthly = t.sum_monthly(ST_tot)
        ST_tot_year = round(ST_tot.sum() / 1000, 2)

        sf_hd_monthly = [100 * sthd / hd for sthd, hd in zip(list(ST_hd_monthly.values()), list(hd_tot_monthly.values()))]
        sf_ts_monthly = [100 * tshd / hd for tshd, hd in zip(list(ST_ts_monthly.values()), list(hd_tot_monthly.values()))]

        sf_tot_year = (ST_hd_year + ST_ts_year) / hd_tot_year * 100
        
        """print ('ST_hd_year', ST_hd_year)
        print ('ST_ts_year', ST_ts_year)
        print ('ST_tot_year', ST_tot_year)

        print ('sf tot', sf_tot_year)"""

        sf_tot_year_new = (ST_tot_year) / hd_tot_year * 100
        
        # bar chart of months
        plt.figure()

        plt.subplot(1,1,1)

        bar1 = plt.bar(range(len(ST_hd_monthly)), sf_hd_monthly, align='center', label='ST to heat demand')

        bar2 = plt.bar(range(len(ST_ts_monthly)), sf_ts_monthly, align='center', bottom=sf_hd_monthly, label='ST to thermal storage')

        plt.xticks(range(len(ST_hd_monthly)), ST_hd_monthly.keys())
        plt.yticks()
        plt.ylabel('Solar Fraction (%)')
        plt.title(f'Yearly Solar Fraction (%): {sf_tot_year:.2f}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def CO2_bar(self):
        # monthly bar chart of CO2 emissions
        # assume PV, HP, ST, TS zero emissions
        # assume 0.27kgCO2/kWh for electricity imported for HP and aux
        # if negative - just means that more renewable electricity is produced than imported for example

        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'CO2_bar_chart.png')

        results = self.results
        timesteps = self.timesteps

        elec_imported = np.zeros(timesteps) # elec imported for HP or aux
        elec_exported = np.zeros(timesteps)
        balance = np.zeros(timesteps)

        for i in range(timesteps):
            elec_imported[i] = results[i]['HP']['elec_import_usage'] + results[i]['heat_demand']['heat_demand'] - results[i]['heat_demand']['HP'] - results[i]['heat_demand']['TS'] - results[i]['ST']['heat_to_heat_demand']
            elec_exported[i] = results[i]['RES']['export']
            balance[i] = elec_imported[i] - elec_exported[i]

        emissions_monthly = t.sum_monthly(0.27 * balance)
        emissions_year = round(0.27 * balance.sum() / 1000, 1)
                       
        # bar chart of months
        plt.figure()

        plt.subplot(1, 1, 1)
        plt.bar(
            range(len(emissions_monthly)),
            list(emissions_monthly.values()),
            align='center')
        plt.xticks(range(len(emissions_monthly)), list(emissions_monthly.keys()))
        plt.yticks()
        plt.ylabel('CO2 Emissions Balance (tCO2)')
        plt.title('Total CO2 Emissions Balance (tCO2): %s' % (emissions_year))


        plt.tight_layout()
        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()

    def import_export_bar(self):
        # monthly bar chart of elec import export balance


        # points to where the pdf will be saved and its name
        fileout = os.path.join(self.folder_path, 'import_export_bar_chart.png')

        results = self.results
        timesteps = self.timesteps

        elec_exported = np.zeros(timesteps) # elec exported - useful for net metering projects
        elec_imported = np.zeros(timesteps) # elec imported for HP or aux
        balance = np.zeros(timesteps) 
        for i in range(timesteps):
            elec_imported[i] = results[i]['HP']['elec_import_usage'] + results[i]['heat_demand']['heat_demand'] - results[i]['heat_demand']['HP'] - results[i]['heat_demand']['TS'] - results[i]['ST']['heat_to_heat_demand']
            elec_exported[i] = results[i]['RES']['export']
            balance[i] = elec_imported[i] - elec_exported[i]



        balance_monthly = t.sum_monthly(balance) #MWh
        balance_year = round(balance.sum() / 1000, 1) #MWh

        # bar chart of months
        plt.figure()

        plt.subplot(1, 1, 1)
        plt.bar(
            range(len(balance_monthly)),
            list(balance_monthly.values()),
            align='center')
        plt.xticks(range(len(balance_monthly)), list(balance_monthly.keys()))
        plt.yticks()
        plt.ylabel('Import-Export Balance (MWh)')
        plt.title('Total Import-Export Balance (MWh): %s' % (balance_year))


        plt.tight_layout()
        plt.savefig(fileout, format='png', dpi=300, bbox_inches='tight')
        plt.close()


class Calcs(object):

    def __init__(self, name, subname, results):

        self.folder_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", name[:-5], subname)
        self.name = name
        self.subname = subname

        self.file_output_pickle = os.path.join(
            os.path.dirname(__file__), '..', 'outputs',
            name[:-5], subname, 'outputs.pkl')
        self.results = results[subname]

        self.myInputs = inputs.Inputs(name, subname)
        # controller inputs
        controller_info = self.myInputs.controller()['controller_info']
        self.timesteps = controller_info['total_timesteps']
        self.first_hour = controller_info['first_hour']

    # technical outputs

    def sum_ST_output(self):
        # calculates solar thermal energy that is used
        # to heat demand or thermal storage

        results = self.results
        timesteps = self.timesteps

        ST_out = np.zeros(timesteps)

        for i in range(timesteps):
            ST_out[i] = results[i]['ST']['heat_to_heat_demand'] + results[i]['ST']['heat_to_TS']

        sum_ST = np.sum(ST_out)

        return sum_ST

    def sum_elec_import(self):
        # calculates total electricity imported for heating purposes for co2 calcs later

        results = self.results
        timesteps = self.timesteps

        elec_imported = np.zeros(timesteps)

        for i in range(timesteps):
            elec_imported[i] = results[i]['HP']['elec_import_usage'] + results[i]['heat_demand']['heat_demand'] - results[i]['heat_demand']['HP'] - results[i]['heat_demand']['TS'] - results[i]['ST']['heat_to_heat_demand']

        sum_elec_imported = np.sum(elec_imported)

        return sum_elec_imported

    def wasted_ST_output(self):
        # calculates solar thermal energy that is wasted - not used

        results = self.results
        timesteps = self.timesteps

        ST_wasted = np.zeros(timesteps)

        for i in range(timesteps):
            ST_wasted[i] = results[i]['ST']['heat_total_output'] - results[i]['ST']['heat_to_heat_demand'] - results[i]['ST']['heat_to_TS']

        wasted_ST = np.sum(ST_wasted)

        return wasted_ST

    def max_ST_output(self):

        results = self.results
        timesteps = self.timesteps

        ST_max = np.zeros(timesteps)

        for i in range(timesteps):
            ST_max[i] = results[i]['ST']['heat_total_output']

        max_ST = np.max(ST_max)

        return max_ST

    def max_heat_pump_output(self):

        results = self.results
        timesteps = self.timesteps

        HPt = np.zeros(timesteps)

        for i in range(timesteps):
            HPt[i] = results[i]['HP']['heat_total_output']

        max_HPt = np.amax(HPt)

        return max_HPt
    
    def sum_heat_demand(self):

        results = self.results
        timesteps = self.timesteps

        hd = np.zeros(timesteps)

        for i in range(timesteps):
            hd[i] = results[i]['heat_demand']['heat_demand']

        sum_hd = np.sum(hd)

        return sum_hd

    def max_heat_demand(self):

        results = self.results
        timesteps = self.timesteps

        hd = np.zeros(timesteps)

        for i in range(timesteps):
            hd[i] = results[i]['heat_demand']['heat_demand']

        max_hd = np.amax(hd)

        return max_hd

    def sum_aux_output(self):

        results = self.results
        timesteps = self.timesteps

        aux = np.zeros(timesteps)

        for i in range(timesteps):
            aux[i] = results[i]['aux']['demand']

        sum_aux = np.sum(aux)

        return sum_aux

    def sum_hp_output(self):

        results = self.results
        timesteps = self.timesteps

        HPt = np.zeros(timesteps)

        for i in range(timesteps):
            HPt[i] = results[i]['HP']['heat_total_output']

        sum_HPt = np.sum(HPt)

        return sum_HPt

    def sum_hp_usage(self):

        results = self.results
        timesteps = self.timesteps

        HPe = np.zeros(timesteps)

        for i in range(timesteps):
            HPe[i] = results[i]['HP']['elec_total_usage']

        sum_HPe = np.sum(HPe)

        return sum_HPe

    def calc_scop(self):

        scop = self.sum_hp_output() / self.sum_hp_usage()
        return scop

    def sum_hd(self):

        results = self.results
        timesteps = self.timesteps

        hd = np.zeros(timesteps)

        for i in range(timesteps):
            hd[i] = results[i]['heat_demand']['heat_demand']

        sum_hd = np.sum(hd)

        return sum_hd

    def sum_ed(self):

        results = self.results
        timesteps = self.timesteps

        ed = np.zeros(timesteps)

        for i in range(timesteps):
            ed[i] = results[i]['elec_demand']['elec_demand']

        sum_ed = np.sum(ed)

        return sum_ed

    def sum_ed_import(self):

        results = self.results
        timesteps = self.timesteps

        ed = np.zeros(timesteps)

        for i in range(timesteps):
            ed[i] = results[i]['grid']['import_for_elec_demand']

        sum_ed = np.sum(ed)

        return sum_ed

    def sum_ed_RES(self):

        results = self.results
        timesteps = self.timesteps

        ed = np.zeros(timesteps)

        for i in range(timesteps):
            ed[i] = results[i]['RES']['elec_demand']

        sum_ed = np.sum(ed)

        return sum_ed

    def sum_import(self):

        results = self.results
        timesteps = self.timesteps

        imp = np.zeros(timesteps)

        for i in range(timesteps):
            imp[i] = results[i]['grid']['total_import']

        sum_imp = np.sum(imp)

        return sum_imp

    def sum_export(self):

        results = self.results
        timesteps = self.timesteps

        exp = np.zeros(timesteps)

        for i in range(timesteps):
            exp[i] = results[i]['grid']['total_export']

        sum_exp = np.sum(exp)

        return sum_exp

    def sum_elec_balance(self):
        
        sum_balance = self.sum_import() - self.sum_export()

        return sum_balance

    def sum_RES(self):

        results = self.results
        timesteps = self.timesteps

        RES = np.zeros(timesteps)

        for i in range(timesteps):
            RES[i] = results[i]['RES']['generation_total']

        sum_RES = np.sum(RES)

        return sum_RES

    # technical KPIs
    
    def CO2_emissions(self):
        # calculates CO2 emissions savings between existing system and system modelled
        # existing system assumed: 60% gas boiler, 30% elec heater, 10% solar thermal
        # solar thermal assumed 0 emissions
        # return tCO2
        # if negative - just means that more renewable electricity is produced than imported for example


        tot_elec_imported = self.sum_elec_balance()
        #tot_heat_demand = self.sum_hd()

        #gas_boiler_emission = 0.2 #kgCO2/kWh
        elec_generation_emission = 0.27 #kgCO2/kWh

        #saving = tot_heat_demand * (0.6 * gas_boiler_emission + 0.3 * elec_generation_emission) - tot_elec_imported * elec_generation_emission
        emission = tot_elec_imported * elec_generation_emission / 1000

        return round(emission, 1)

    def HP_size_ratio(self):

        # look over whole period
        # find highest heat pump thermal output
        # find highest heat demand
        # max hd / max hpto = plant size change
        # return percentage

        max_HPt = self.max_heat_pump_output()
        max_hd = self.max_heat_demand()

        return round(max_HPt / max_hd, 2)

    def HP_utilisation(self):

        sum_hp = self.sum_hp_output()
        sum_hd = self.sum_hd()

        ratio = sum_hp / sum_hd

        return round(ratio, 2) * 100

    def RES_self_consumption(self):

        _sum_RES = self.sum_RES()
        _sum_export = self.sum_export()

        if _sum_RES > 0:
            ratio = 1 - _sum_export / float(_sum_RES)
        else:
            ratio = 0

        return ratio * 100

    def total_RES_self_consumption(self):

        _sum_RES = self.sum_RES() - self.sum_export()

        return _sum_RES

    def grid_RES_used(self):

        name = self.name
        subname = self.subname

        results = self.results
        timesteps = self.timesteps
        # what is the discount price?
        grid_inputs = self.myInputs.grid()
        export = grid_inputs['export']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        variable_periods_year = grid_inputs['variable_periods_year']
        premium = grid_inputs['wm_info']['premium']
        maximum = grid_inputs['wm_info']['maximum']
        lower_percent = grid_inputs['ppa_info']['lower_percent']
        higher_percent = grid_inputs['ppa_info']['higher_percent']
        lower_penalty = grid_inputs['ppa_info']['lower_penalty']
        higher_discount = grid_inputs['ppa_info']['higher_discount']

        if tariff_choice == 'Flat rates':

            return 0.

        elif tariff_choice == 'Variable periods':

            return 0.

        elif tariff_choice == 'Time of use - WM':

            return 0.

        elif tariff_choice == 'Time of use - PPA + FR':

            fr = grid_inputs['flat_rates']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                flat_rate=fr, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + WM':

            twm = grid_inputs['wholesale_market']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                wholesale_market=twm, premium=premium,
                maximum=maximum, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + VP':

            vp = grid_inputs['variable_periods']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                variable_periods=vp,
                variable_periods_year=variable_periods_year,
                lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        wind_farm = myGrid.wind_farm_info()
        higher_band = wind_farm['higher_band']
        power = wind_farm['power']

        grid_RES_import = np.zeros(timesteps)
        for i in range(timesteps):
            if power[i] >= higher_band:
                grid_RES_import[i] = (
                    results[i]['grid']['total_import'])
            else:
                grid_RES_import[i] = 0.

        return np.sum(grid_RES_import)

    def heat_grid_RES_used(self):

        name = self.name
        subname = self.subname

        results = self.results
        timesteps = self.timesteps
        # what is the discount price?
        grid_inputs = self.myInputs.grid()
        export = grid_inputs['export']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        variable_periods_year = grid_inputs['variable_periods_year']
        premium = grid_inputs['wm_info']['premium']
        maximum = grid_inputs['wm_info']['maximum']
        lower_percent = grid_inputs['ppa_info']['lower_percent']
        higher_percent = grid_inputs['ppa_info']['higher_percent']
        lower_penalty = grid_inputs['ppa_info']['lower_penalty']
        higher_discount = grid_inputs['ppa_info']['higher_discount']

        if tariff_choice == 'Flat rates':

            return 0.

        elif tariff_choice == 'Variable periods':

            return 0.

        elif tariff_choice == 'Time of use - WM':

            return 0.

        elif tariff_choice == 'Time of use - PPA + FR':

            fr = grid_inputs['flat_rates']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                flat_rate=fr, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + WM':

            twm = grid_inputs['wholesale_market']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                wholesale_market=twm, premium=premium,
                maximum=maximum, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + VP':

            vp = grid_inputs['variable_periods']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                variable_periods=vp,
                variable_periods_year=variable_periods_year,
                lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        wind_farm = myGrid.wind_farm_info()
        higher_band = wind_farm['higher_band']
        power = wind_farm['power']

        grid_RES_import = np.zeros(timesteps)
        for i in range(timesteps):
            if power[i] >= higher_band:
                grid_RES_import[i] = (
                    results[i]['aux']['demand'] - results[i]['RES']['aux'] +
                    results[i]['HP']['elec_import_usage'])
            else:
                grid_RES_import[i] = 0.

        return np.sum(grid_RES_import)

    def heat_from_grid_RES(self):

        name = self.name
        subname = self.subname

        results = self.results
        timesteps = self.timesteps
        # what is the discount price?
        grid_inputs = self.myInputs.grid()
        export = grid_inputs['export']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        variable_periods_year = grid_inputs['variable_periods_year']
        premium = grid_inputs['wm_info']['premium']
        maximum = grid_inputs['wm_info']['maximum']
        lower_percent = grid_inputs['ppa_info']['lower_percent']
        higher_percent = grid_inputs['ppa_info']['higher_percent']
        lower_penalty = grid_inputs['ppa_info']['lower_penalty']
        higher_discount = grid_inputs['ppa_info']['higher_discount']

        if tariff_choice == 'Flat rates':

            return 0.

        elif tariff_choice == 'Variable periods':

            return 0.

        elif tariff_choice == 'Time of use - WM':

            return 0.

        elif tariff_choice == 'Time of use - PPA + FR':

            fr = grid_inputs['flat_rates']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                flat_rate=fr, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + WM':

            twm = grid_inputs['wholesale_market']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                wholesale_market=twm, premium=premium,
                maximum=maximum, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + VP':

            vp = grid_inputs['variable_periods']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                variable_periods=vp,
                variable_periods_year=variable_periods_year,
                lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        wind_farm = myGrid.wind_farm_info()
        higher_band = wind_farm['higher_band']
        power = wind_farm['power']

        grid_RES_import = np.zeros(timesteps)
        for i in range(timesteps):
            if power[i] >= higher_band:
                grid_RES_import[i] = (
                    results[i]['aux']['demand'] - results[i]['RES']['aux'] +
                    results[i]['HP']['elec_import_usage'] *
                    results[i]['HP']['cop'])
            else:
                grid_RES_import[i] = 0.

        return np.sum(grid_RES_import)

    def HP_from_grid_RES(self):

        name = self.name
        subname = self.subname

        results = self.results
        timesteps = self.timesteps
        # what is the discount price?
        grid_inputs = self.myInputs.grid()
        export = grid_inputs['export']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        tariff_choice = grid_inputs['tariff_choice']
        balancing_mechanism = grid_inputs['balancing_mechanism']
        grid_services = grid_inputs['grid_services']
        variable_periods_year = grid_inputs['variable_periods_year']
        premium = grid_inputs['wm_info']['premium']
        maximum = grid_inputs['wm_info']['maximum']
        lower_percent = grid_inputs['ppa_info']['lower_percent']
        higher_percent = grid_inputs['ppa_info']['higher_percent']
        lower_penalty = grid_inputs['ppa_info']['lower_penalty']
        higher_discount = grid_inputs['ppa_info']['higher_discount']

        if tariff_choice == 'Flat rates':

            return 0.

        elif tariff_choice == 'Variable periods':

            return 0.

        elif tariff_choice == 'Time of use - WM':

            return 0.

        elif tariff_choice == 'Time of use - PPA + FR':

            fr = grid_inputs['flat_rates']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                flat_rate=fr, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + WM':

            twm = grid_inputs['wholesale_market']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                wholesale_market=twm, premium=premium,
                maximum=maximum, lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        elif tariff_choice == 'Time of use - PPA + VP':

            vp = grid_inputs['variable_periods']
            myGrid = grid.Grid(
                name, subname, export,
                tariff_choice, balancing_mechanism, grid_services,
                variable_periods=vp,
                variable_periods_year=variable_periods_year,
                lower_percent=lower_percent,
                higher_percent=higher_percent, higher_discount=higher_discount,
                lower_penalty=lower_penalty)

        wind_farm = myGrid.wind_farm_info()
        higher_band = wind_farm['higher_band']
        power = wind_farm['power']

        grid_RES_import = np.zeros(timesteps)
        for i in range(timesteps):
            if power[i] >= higher_band:
                grid_RES_import[i] = (
                    results[i]['HP']['elec_import_usage'] *
                    results[i]['HP']['cop'])
            else:
                grid_RES_import[i] = 0.

        return np.sum(grid_RES_import)

    def heat_from_local_RES(self):

        results = self.results
        timesteps = self.timesteps

        aux_res = np.zeros(timesteps)
        hp_res = np.zeros(timesteps)

        for i in range(timesteps):
            aux_res[i] = results[i]['RES']['aux']
            hp_res[i] = (
                results[i]['HP']['elec_RES_usage'] *
                results[i]['HP']['cop'])

        tot = np.sum(aux_res) + np.sum(hp_res) + self.sum_ST_output()

        return tot

    def HP_from_local_RES(self):

        results = self.results
        timesteps = self.timesteps

        hp_res = np.zeros(timesteps)

        for i in range(timesteps):
            hp_res[i] = (
                results[i]['HP']['elec_RES_usage'] *
                results[i]['HP']['cop'])

        tot = np.sum(hp_res)

        return tot

    def HP_total_RES_used(self):
        tot = (
            self.HP_from_local_RES() +
            self.HP_from_grid_RES())
        return tot

    def total_RES_used(self):
        tot = self.grid_RES_used() + self.total_RES_self_consumption()
        return tot

    def heat_total_RES_used(self):

        tot = (
            self.heat_grid_RES_used() +
            self.total_RES_self_consumption() -
            self.sum_ed_RES())
        return tot

    def demand_met_RES(self):
        # sum the heat pump elec usage, demand from aux usage,
        # and elec demand usage
        # percentage of this from RES used

        demand = self.sum_hp_usage() + self.sum_aux_output() + self.sum_ed()
        RES_used = self.total_RES_used()

        dmRES = (1 - (demand - RES_used) / demand) * 100
        return dmRES

    def heat_met_RES(self):
        # sum the heat pump output, aux output and solar thermal output 
        # percentage of this from RES used

        demand = self.sum_hp_output() + self.sum_aux_output() + self.sum_ST_output()
        # demand = self.sum_hd()
        RES_used = self.heat_from_local_RES() + self.heat_from_grid_RES()

        
        dmRES = RES_used / demand * 100
        return dmRES

    def HP_met_RES(self):
        # sum the heat pump elec usage, demand from aux usage,
        # percentage of this from RES used

        demand = self.sum_hp_output()
        RES_used = self.HP_from_local_RES() + self.HP_from_grid_RES()

        dmRES = RES_used / demand * 100
        return dmRES

    def days_storage_content(self):

        # factor for counting for when not simulating whole year
        f = self.timesteps / 8760
        av_day_demand = self.sum_hd() / (365 * f)

        str = self.subname
        str = str.replace("_", " ")
        t = [int(s) for s in str.split() if s.isdigit()]
        capacity = t[1]
        # presuming a delta t of 20
        energy = capacity * 4.18 * 40 / 3600

        day_content = energy / av_day_demand
        return day_content

    def heat_met_ST(self):
        # % of total heat demand met by solar thermal
        # 

        demand = self.sum_hp_output() + self.sum_aux_output() + self.sum_ST_output()
        #RES_used = self.HP_from_local_RES() + self.HP_from_grid_RES()

        dmST = self.sum_ST_output() / demand * 100
        return dmST

    def year_sf(self):
        # solar fraction

        sf = self.sum_ST_output() / self.sum_heat_demand() * 100

        return sf

    def year_wasted_ST(self):
        # solar fraction
        # % of total ST energy that can be produced by ST panels

        wasted_ST = self.wasted_ST_output() / (self.sum_ST_output() + self.wasted_ST_output()) * 100

        return wasted_ST

    # economic KPIs

    def RHI_income(self):

        HP_sum = self.sum_hp_output()
        HPe_sum = self.sum_hp_usage()
        HP_eligble = HP_sum - HPe_sum

        str = self.subname
        str = str.replace("_", " ")
        t = [int(s) for s in str.split() if s.isdigit()]
        HP_capacity = t[0]
        RHI_info = self.myInputs.RHI()

        if RHI_info['tariff_type'] == 'Fixed':
            year_income = HP_eligble * RHI_info['fixed_rate'] / 100.

        elif RHI_info['tariff_type'] == 'Tiered':
            tier_1_output = HP_capacity * 1314
            if HP_eligble <= tier_1_output:
                year_income = HP_eligble * RHI_info['tier_1'] / 100.
            elif HP_eligble > tier_1_output:
                year_income = (
                    (tier_1_output * RHI_info['tier_1'] +
                     (HP_eligble - tier_1_output) * RHI_info['tier_2']) /
                    100.)

        if RHI_info['RHI_type'] == 'Domestic':
            years = 7.
        elif RHI_info['RHI_type'] == 'Non-domestic':
            years = 20.

        # units are pounds
        total_income = year_income * years

        return {'year_income': year_income, 'total_income': total_income}

    def capital_cost(self):
        # return in thousands EUR
        # pound / kW
        # hp_capex = 550
        # hp_capex = 600
        # pound / L
        # ts_capex = 3

        def hp_calc(hp_size):
            # danish energy agency tech data
            # return in EUR
            # heat pump size converted from kW to MW

            if hp_size <= 0.0:
                return 0.0
            return 12460 * (hp_size / 1000) ** 2 + 705159 * (hp_size / 1000) + 802381

        def ts_calc(ts_size):
            # see the excel sheet for the line fitting
            # 'thermal_storage_costs.xlsx'
            # return in EUR / metre cubed
            # danish energy agency tech data
            if ts_size <= 0.0:
                return 0.0
            return 7450 * ((ts_size / 1000.0) ** (-0.47))

        # function used to be: 7479.1 * ((ts_size / 1000.0) ** (-0.501))

        def st_calc(collector_area):
            # danish energy agency tech data
            # return in EUR

            if collector_area <= 0.0:
                return 0.0
            return 167.73 * collector_area + 263320        
        
        def pv_calc(self):
            # danish energy agency tech data
            # return in EUR

            PV = np.zeros(self.timesteps)
            for i in range(self.timesteps):
                PV[i] = self.results[i]['RES']['PV']
                PV_year = round(PV.sum() / 1000, 2) # this is in MWh
            
            total_PV_power = PV_year / 8760
            if total_PV_power <= 0.0:
                return 0.0
            return 0.56 * total_PV_power * 10 ** 6
        


        str = self.subname
        str = str.replace("_", " ")
        t = [int(s) for s in str.split() if s.isdigit()]
        print(t)
        hp_size = t[0]
        ts_size = t[1]
        #st_size = t[2]

        
        # capex in EUR
        capex = st_calc(self.myInputs.ST_model()['collector_area']) + hp_calc(hp_size) + ts_size * ts_calc(ts_size) / 1000 + pv_calc(self)
        # capex = hp_size * hp_capex + ts_size * ts_capex

        
        return capex / 1000.

    def operating_cost(self):

        # operating cost includes covering
        # electrical and thermal demand

        results = self.results
        timesteps = self.timesteps

        cashflow = np.zeros(timesteps)
        aux_cost = np.zeros(timesteps)

        for i in range(timesteps):
            cashflow[i] = results[i]['grid']['cashflow']
            aux_cost[i] = results[i]['aux']['cost'] / 1000.

        # different for electric aux/non electric aux
        # with electric aux cost included in import

        if self.myInputs.aux()['fuel'] == 'Electric':
            opex = np.sum(-cashflow)
        else:
            opex = np.sum(-cashflow) + np.sum(aux_cost)

        return opex / 1000

    def cost_of_heat(self):

        results = self.results
        timesteps = self.timesteps

        heat_cost = np.zeros(timesteps)

        if self.myInputs.aux()['fuel'] == 'Electric':
            for i in range(timesteps):
                heat_cost[i] = (
                    (results[i]['grid']['total_import'] -
                     results[i]['grid']['import_for_elec_demand']) *
                    results[i]['grid']['import_price'] / 1000.)

        else:
            for i in range(timesteps):
                heat_cost[i] = (
                    ((results[i]['grid']['total_import'] -
                     results[i]['grid']['import_for_elec_demand']) *
                     results[i]['grid']['import_price'] / 1000.) +
                    results[i]['aux']['cost'] / 1000.)

        # total heat output from aux and heatpump
        aux_tot = self.sum_aux_output()
        hp_tot = self.sum_hp_output()
        heat_cost_tot = np.sum(heat_cost)
        RHI = self.RHI_income()['year_income']

        COH = (heat_cost_tot - RHI) / (aux_tot + hp_tot)

        return COH

    def levelised_cost_of_heat(self):

        results = self.results
        timesteps = self.timesteps

        heat_cost = np.zeros(timesteps)

        if self.myInputs.aux()['fuel'] == 'Electric':
            for i in range(timesteps):
                heat_cost[i] = (
                    (results[i]['grid']['total_import'] -
                     results[i]['grid']['import_for_elec_demand']) *
                    results[i]['grid']['import_price'] / 1000.)

        else:
            for i in range(timesteps):
                heat_cost[i] = (
                    (results[i]['grid']['total_import'] -
                     results[i]['grid']['import_for_elec_demand']) *
                    results[i]['grid']['import_price'] / 1000. +
                    results[i]['aux']['cost'] / 1000)

        # capital cost + operating cost divided by total energy output
        capex = self.capital_cost() * 1000
        heat_cost_tot = np.sum(heat_cost)
        RHI = self.RHI_income()['total_income']

        # total heat output from aux and heatpump
        hd = self.sum_hd()

        # factor for counting for when not simulating whole year
        f = 8760 / self.timesteps

        # assuming 20 year life
        LCOH = (
            ((capex + heat_cost_tot * f * 20) - RHI) /
            ((hd) * f * 20))

        return LCOH

    def levelised_cost_of_energy(self):

        results = self.results
        timesteps = self.timesteps

        cashflow = np.zeros(timesteps)

        if self.myInputs.aux()['fuel'] == 'Electric':
            for i in range(timesteps):
                cashflow[i] = (
                    (-results[i]['grid']['cashflow'] / 1000.))

        else:
            for i in range(timesteps):
                cashflow[i] = (
                    (-results[i]['grid']['cashflow'] / 1000. +
                     results[i]['aux']['cost'] / 1000))

        # capital cost + operating cost divided by total energy output
        capex = self.capital_cost() * 1000
        energy_cost_tot = np.sum(cashflow) * 1000
        RHI = self.RHI_income()['total_income']

        # total energy output from heat pump, elec demand met and aux
        heat_dem = self.sum_hd()
        elec_dem = self.sum_ed()

        # factor for counting for when not simulating whole year
        f = 8760 / self.timesteps

        # assuming 20 year life
        LCOE = (
            ((capex + energy_cost_tot * f * 20) - RHI) /
            ((heat_dem + elec_dem) * f * 20))

        return LCOE

    def cost_elec(self):

        results = self.results
        timesteps = self.timesteps

        elec_cost = np.zeros(timesteps)

        for i in range(timesteps):
            elec_cost[i] = (
                (-results[i]['grid']['cashflow'] / 1000.))

        return np.sum(elec_cost)

    def lifetime_cost(self):
        
        # €/community member

        results = self.results
        timesteps = self.timesteps

        energy_cost = np.zeros(timesteps)

        if self.myInputs.aux()['fuel'] == 'Electric':
            for i in range(timesteps):
                energy_cost[i] = (
                    (-results[i]['grid']['cashflow'] / 1000.))

        else:
            for i in range(timesteps):
                energy_cost[i] = (
                    (-results[i]['grid']['cashflow'] / 1000. +
                     results[i]['aux']['cost'] / 1000))

        # capital cost + operating cost divided by total energy output
        capex = self.capital_cost()
        energy_cost_tot = np.sum(energy_cost)

        # factor for counting for when not simulating whole year
        f = 8760 / self.timesteps

        # assuming 20 year life
        life_cost = (
            (capex + energy_cost_tot * f * 20))

        return life_cost/60


class ThreeDPlots(object):

    def __init__(self, name):

        # folder path name is in main name alongside parametric solutions
        self.folder_path = os.path.join(
            os.path.dirname(__file__), "..", "outputs", name[:-5], 'KPIs')
        self.name = name

        # creates a folder for keeping all the
        # pickle input files as saved from the excel file
        if os.path.isdir(self.folder_path) is False:
            os.mkdir(self.folder_path)

        elif os.path.isdir(self.folder_path) is True:
            shutil.rmtree(self.folder_path)
            os.mkdir(self.folder_path)
        # read in set of parameters from input
        file1 = os.path.join(
            os.path.dirname(__file__), "..", "inputs", name[:-5],
            "inputs.pkl")
        self.input = pd.read_pickle(file1)
        pa = self.input['parametric_analysis']
        # list set of heat pump sizes
        hp_sizes = []
        # if no step then only one size looked at
        if pa['hp_min'] == pa['hp_max']:
            hp_sizes.append(pa['hp_min'])
        else:
            for i in range(pa['hp_min'], pa['hp_max'] + pa['hp_step'], pa['hp_step']):
                hp_sizes.append(i)

        # list sizes of ts
        ts_sizes = []
        # if no step then only one size looked at
        if pa['ts_min'] == pa['ts_max']:
            ts_sizes.append(pa['ts_min'])
        else:
            for i in range(pa['ts_min'], pa['ts_max'] + pa['ts_step'], pa['ts_step']):
                ts_sizes.append(i)

        # list sizes of st
        st_sizes = []
        # if no step then only one size looked at
        if pa['st_min'] == pa['st_max']:
            st_sizes.append(pa['st_min'])
        else:
            for i in range(pa['st_min'], pa['st_max'] + pa['st_step'], pa['st_step']):
                st_sizes.append(i)    

        combos = []
        for i in range(len(hp_sizes)):
            for j in range(len(ts_sizes)):
                for k in range(len(st_sizes)):  
                    combos.append([hp_sizes[i], ts_sizes[j], st_sizes[k]])
        self.combos = combos

        # strings for all combos to READ OUTPUTS
        results = {}
        subnames = []
        #print('combos', self.combos)
        for i in range(len(combos)):
            subname = 'hp_' + str(combos[i][0]) + '_ts_' + str(combos[i][1]) + '_st_' + str(combos[i][2])
            subnames.append(subname)
            # read pickle output file
            file_output_pickle = os.path.join(
                os.path.dirname(__file__), '..', 'outputs',
                self.name[:-5], subname, 'outputs.pkl')
            results[subname] = pd.read_pickle(file_output_pickle)
        self.results = results
        self.subnames = subnames

    def heat_pump_sizes_x(self):

        hp_sizes = []
        for i in range(len(self.combos)):
            hp_sizes.append(self.combos[i][0])
        return hp_sizes

    def thermal_store_sizes_y(self):

        ts_sizes = []
        for i in range(len(self.combos)):
            ts_sizes.append(self.combos[i][1] / 1000.)
        return ts_sizes
    
    def solar_thermal_sizes(self):

        st_sizes = []
        for i in range(len(self.combos)):
            st_sizes.append(self.combos[i][2])
        return st_sizes
    

    def plot_carbon_emissions(self):

        emissions = []

        for x in range(len(self.subnames)):
            _emissions = Calcs(
                self.name, self.subnames[x],
                self.results).CO2_emissions()
            emissions.append(_emissions)

        z = emissions

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        y = self.thermal_store_sizes_y()
        x = self.solar_thermal_sizes()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        ax.set_xlabel('Soalr thermal sizes ($m^2$)', fontsize=10)
        ax.set_ylabel('Thermal store sizes ($m^3$)', fontsize=10)
        ax.set_zlabel(u'Carbon emissions (tCO2)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        # ax.invert_yaxis()

        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'carbon.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()



    def plot_wasted_heat_from_ST(self):

        ST_wasted = []

        for x in range(len(self.subnames)):
            _ST_wasted = Calcs(
                self.name, self.subnames[x],
                self.results).year_wasted_ST()
            ST_wasted.append(_ST_wasted)


        y = self.thermal_store_sizes_y()
        z = ST_wasted

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.solar_thermal_sizes()
        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_xlabel('Solar thermal sizes ($m^2$)', fontsize=10)
        ax.set_zlabel('Yearly wasted ST energy (%)', fontsize=10)
        ax.set_ylabel('Thermal store sizes ($m^3$)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_zlim(0)
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'wasted_sf_ST_heat.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_heat_from_ST(self):

        ST_used = []

        for x in range(len(self.subnames)):
            _ST_used = Calcs(
                self.name, self.subnames[x],
                self.results).year_sf()
            ST_used.append(_ST_used)

        z = ST_used

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.solar_thermal_sizes()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_xlabel('Solar thermal sizes ($m^2$)', fontsize=10)
        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_zlabel('Yearly solar fraction (%)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        ax.invert_xaxis()
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'heat_from_ST.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_heat_from_RES(self):

        RES_used = []

        for x in range(len(self.subnames)):
            _RES_used = Calcs(
                self.name, self.subnames[x],
                self.results).heat_met_RES()
            RES_used.append(_RES_used)

        z = RES_used


        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.solar_thermal_sizes()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_xlabel('Solar thermal sizes ($m^2$)', fontsize=10)
        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_zlabel('Heat demand from RES (%)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        ax.invert_xaxis()
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'heat_from_RES.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()
        
    def plot_capital_cost_st(self):

        capital_cost = []

        for x in range(len(self.subnames)):
            _capital_cost = Calcs(
                self.name, self.subnames[x],
                self.results).capital_cost()
            capital_cost.append(_capital_cost)

        z = capital_cost

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.solar_thermal_sizes()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        ax.invert_xaxis()
        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_xlabel('Solar thermal sizes ($m^2$)', fontsize=10)
        ax.set_zlabel(u'Capital cost (k\u20AC)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'capital_cost_st.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()


    def plot_opex(self):

        opex = []

        for x in range(len(self.subnames)):
            _opex = Calcs(
                self.name, self.subnames[x],
                self.results).operating_cost()
            opex.append(_opex)

        z = opex

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        y = self.heat_pump_sizes_x()
        x = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        ax.set_xlabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_ylabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_zlabel(u'Operational cost (k\u20AC)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        # ax.invert_yaxis()

        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'operating_cost.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_RES(self):

        RES_used = []

        for x in range(len(self.subnames)):
            _RES_used = Calcs(
                self.name, self.subnames[x],
                self.results).RES_self_consumption()
            RES_used.append(_RES_used)

        z = RES_used

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.heat_pump_sizes_x()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_xlabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_zlabel('RES self-consumption (%)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'RES_self_consumption.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_HP_size_ratio(self):

        HP_size_ratio = []

        for x in range(len(self.subnames)):
            _HP_size_ratio = Calcs(
                self.name, self.subnames[x],
                self.results).HP_size_ratio()
            HP_size_ratio.append(_HP_size_ratio)

        z = HP_size_ratio

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.heat_pump_sizes_x()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        #ax.invert_xaxis()
        ax.set_xlabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_xlabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_zlabel('HP size ratio', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'HP_size_ratio.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_HP_utilisation(self):

        HP_utilisation = []

        for x in range(len(self.subnames)):
            _HP_utilisation = Calcs(
                self.name, self.subnames[x],
                self.results).HP_utilisation()
            HP_utilisation.append(_HP_utilisation)

        z = HP_utilisation

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.heat_pump_sizes_x()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        #ax.invert_xaxis()

        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_xlabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_zlabel('HP utilisation (%)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'HP_utilisation.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_capital_cost_hp(self):

        capital_cost = []

        for x in range(len(self.subnames)):
            _capital_cost = Calcs(
                self.name, self.subnames[x],
                self.results).capital_cost()
            capital_cost.append(_capital_cost)

        z = capital_cost

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.heat_pump_sizes_x()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)
        #ax.invert_xaxis()
        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_xlabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_zlabel(u'Capital cost (k\u20AC)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'capital_cost_hp.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_COH(self):

        cost_of_heat = []

        for x in range(len(self.subnames)):
            _cost_of_heat = Calcs(
                self.name, self.subnames[x],
                self.results).cost_of_heat()
            cost_of_heat.append(_cost_of_heat)

        z = cost_of_heat

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        y = self.heat_pump_sizes_x()
        x = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_xlabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_ylabel('Heat pump sizes (kW)', fontsize=10)
        ax.set_zlabel(u'Cost of heat (\u20AC/kWh)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        # ax.invert_yaxis()
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'cost_of_heat.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def plot_LCOH(self):

        levelised_cost_of_heat = []

        for x in range(len(self.subnames)):
            _levelised_cost_of_heat = Calcs(
                self.name, self.subnames[x],
                self.results).levelised_cost_of_heat()
            levelised_cost_of_heat.append(_levelised_cost_of_heat)

        z = levelised_cost_of_heat

        plt.style.use('classic')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = self.solar_thermal_sizes()
        y = self.thermal_store_sizes_y()

        ax.plot_trisurf(x, y, z, linewidth=0, antialiased=False,
                        cmap=plt.cm.cividis)

        ax.set_ylabel('Thermal storage capacity ($m^3$)', fontsize=10)
        ax.set_xlabel('Solar thermal sizes ($m^3$)', fontsize=10)
        ax.set_zlabel(u'Levelised cost of heat (\u20AC/kWh)', fontsize=10)
        ax.xaxis.labelpad = 15
        ax.yaxis.labelpad = 15
        ax.zaxis.labelpad = 15
        ax.invert_xaxis()
        # plt.show()
        plt.tight_layout()

        fileout = os.path.join(self.folder_path, 'levelised_cost_of_heat.png')
        plt.savefig(fileout, format='png', dpi=300)
        plt.close()

    def KPIs_to_csv(self):

        hp_sizes = self.heat_pump_sizes_x()
        ts_sizes = self.thermal_store_sizes_y()
        st_sizes = self.solar_thermal_sizes()

        opex = np.zeros(len(self.subnames))
        RES_used = np.zeros(len(self.subnames))
        total_RES_self_consumption = np.zeros(len(self.subnames))
        grid_RES_used = np.zeros(len(self.subnames))
        total_RES_used = np.zeros(len(self.subnames))
        HP_size_ratio = np.zeros(len(self.subnames))
        HP_utilisation = np.zeros(len(self.subnames))
        capital_cost = np.zeros(len(self.subnames))
        cost_of_heat = np.zeros(len(self.subnames))
        levelised_cost_of_heat = np.zeros(len(self.subnames))
        levelised_cost_of_energy = np.zeros(len(self.subnames))
        cost_elec = np.zeros(len(self.subnames))
        lifetime_cost = np.zeros(len(self.subnames))
        sum_hp_output = np.zeros(len(self.subnames))
        sum_hp_usage = np.zeros(len(self.subnames))
        scop = np.zeros(len(self.subnames))
        sum_aux_output = np.zeros(len(self.subnames))
        sum_ed_import = np.zeros(len(self.subnames))
        sum_RES = np.zeros(len(self.subnames))
        sum_import = np.zeros(len(self.subnames))
        sum_export = np.zeros(len(self.subnames))
        demand_met_RES = np.zeros(len(self.subnames))
        HP_met_RES = np.zeros(len(self.subnames))
        days_storage_content = np.zeros(len(self.subnames))
        heat_met_RES = np.zeros(len(self.subnames))
        sum_ST_output = np.zeros(len(self.subnames))
        heat_met_ST = np.zeros(len(self.subnames))
        CO2_emissions = np.zeros(len(self.subnames))
        wasted_ST_output = np.zeros(len(self.subnames))
        sum_elec_balance = np.zeros(len(self.subnames))

        for x in range(len(self.subnames)):

            myCalcs = Calcs(self.name, self.subnames[x], self.results)

            opex[x] = myCalcs.operating_cost()
            RES_used[x] = myCalcs.RES_self_consumption()
            total_RES_self_consumption[x] = myCalcs.total_RES_self_consumption()
            grid_RES_used[x] = myCalcs.grid_RES_used()
            total_RES_used[x] = myCalcs.total_RES_used()
            HP_size_ratio[x] = myCalcs.HP_size_ratio()
            HP_utilisation[x] = myCalcs.HP_utilisation()
            capital_cost[x] = myCalcs.capital_cost()
            cost_of_heat[x] = myCalcs.cost_of_heat()
            levelised_cost_of_heat[x] = myCalcs.levelised_cost_of_heat()
            levelised_cost_of_energy[x] = myCalcs.levelised_cost_of_energy()
            cost_elec[x] = myCalcs.cost_elec()
            lifetime_cost[x] = myCalcs.lifetime_cost()
            sum_hp_output[x] = myCalcs.sum_hp_output()
            sum_hp_usage[x] = myCalcs.sum_hp_usage()
            scop[x] = myCalcs.calc_scop()
            sum_aux_output[x] = myCalcs.sum_aux_output()
            sum_ed_import[x] = myCalcs.sum_ed_import()
            sum_RES[x] = myCalcs.sum_RES()
            sum_import[x] = myCalcs.sum_import()
            sum_export[x] = myCalcs.sum_export()
            demand_met_RES[x] = myCalcs.demand_met_RES()
            HP_met_RES[x] = myCalcs.HP_met_RES()
            days_storage_content[x] = myCalcs.days_storage_content()
            heat_met_RES[x] = myCalcs.heat_met_RES()
            sum_ST_output[x] = myCalcs.sum_ST_output()
            heat_met_ST[x] = myCalcs.heat_met_ST()
            CO2_emissions[x] = myCalcs.CO2_emissions()
            wasted_ST_output[x] = myCalcs.wasted_ST_output()
            sum_elec_balance[x] = myCalcs.sum_elec_balance()

        print(hp_sizes)

        economic_data = np.array(
            [hp_sizes, ts_sizes, st_sizes,
             capital_cost, opex, cost_of_heat,
             cost_elec, lifetime_cost,
             levelised_cost_of_heat,
             levelised_cost_of_energy])
        df = pd.DataFrame(
            economic_data)
        df = df.transpose()
        df.columns = ['hp_sizes', 'ts_sizes', 'st_sizes',
                      'capital_cost', 'opex', 'cost_of_heat',
                      'cost_elec', 'lifetime_cost',
                      'levelised_cost_of_heat',
                      'levelised_cost_of_energy'
                      ]

        pickle_name = 'KPI_economic_' + self.name[:-5] + '.pkl'
        pickleout = os.path.join(self.folder_path, pickle_name)
        with open(pickleout, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file_name = 'KPI_economic_' + self.name[:-5] + '.csv'
        fileout = os.path.join(self.folder_path, file_name)
        df.to_csv(fileout, index=False)

        technical_data = np.array(
            [hp_sizes, ts_sizes, st_sizes, RES_used,
             total_RES_self_consumption,
             grid_RES_used, total_RES_used,
             heat_met_RES, demand_met_RES,
             HP_met_RES, heat_met_ST,
             HP_size_ratio, HP_utilisation,
             days_storage_content, CO2_emissions])
        df = pd.DataFrame(
            technical_data)
        df = df.transpose()
        df.columns = ['hp_sizes', 'ts_sizes', 'st_sizes', 'local_RES_used',
                      'total_RES_self_consumption',
                      'grid_RES_used', 'total_RES_used',
                      'heat_met_RES', 'demand_met_RES',
                      'HP_met_RES', 'heat_met_ST',
                      'HP_size_ratio', 'HP_utilisation',
                      'days_storage_content', 'CO2_emissions'
                      ]

        pickle_name = 'KPI_technical_' + self.name[:-5] + '.pkl'
        pickleout = os.path.join(self.folder_path, pickle_name)
        with open(pickleout, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file_name = 'KPI_technical_' + self.name[:-5] + '.csv'
        fileout = os.path.join(self.folder_path, file_name)
        df.to_csv(fileout, index=False)

        output = np.array(
            [hp_sizes, ts_sizes, st_sizes,
             sum_ST_output,
             sum_hp_output, sum_hp_usage, scop,
             sum_aux_output, sum_ed_import, sum_RES,
             sum_import, sum_export, wasted_ST_output, sum_elec_balance])
        df = pd.DataFrame(
            output)
        df = df.transpose()
        df.columns = ['hp_sizes', 'ts_sizes', 'st_sizes',
                      'sum_ST_output',
                      'sum_hp_output', 'sum_hp_usage', 'scop',
                      'sum_aux_output', 'sum_ed_import', 'sum_RES',
                      'sum_import', 'sum_export', 'wasted_ST_output', 'sum_elec_balance'
                      ]

        pickle_name = 'output_' + self.name[:-5] + '.pkl'
        pickleout = os.path.join(self.folder_path, pickle_name)
        with open(pickleout, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file_name = 'output_' + self.name[:-5] + '.csv'
        fileout = os.path.join(self.folder_path, file_name)
        df.to_csv(fileout, index=False)
