import pandas as pd

def get_retailrate(folder,start,end,city,market_file):

    #Cacluclate retail arte
    df_slack = pd.read_csv(folder+'/load_node_149.csv',skiprows=range(8))
    df_slack['# timestamp'] = df_slack['# timestamp'].map(lambda x: str(x)[:-4])
    df_slack = df_slack.iloc[:-1]
    df_slack['# timestamp'] = pd.to_datetime(df_slack['# timestamp'])
    df_slack.set_index('# timestamp',inplace=True)
    df_slack = df_slack.loc[start:end]
    df_slack = df_slack/1000 #kW

    df_WS = pd.read_csv('glm_generation_'+city+'/'+market_file,parse_dates=[0])
    df_WS.rename(columns={'Unnamed: 0':'timestamp'},inplace=True)
    df_WS.set_index('timestamp',inplace=True)
    df_WS = df_WS.loc[start:end]

    df_WS['system_load'] = df_slack['measured_real_power']
    supply_wlosses = (df_WS['system_load']/1000./12.).sum() # MWh
    df_WS['supply_cost'] = df_WS['system_load']/1000.*df_WS['RT']/12.
    supply_cost_wlosses = df_WS['supply_cost'].sum()

    df_total_load = pd.read_csv(folder+'/total_load_all.csv',skiprows=range(8)) #in kW
    df_total_load['# timestamp'] = df_total_load['# timestamp'].map(lambda x: str(x)[:-4])
    df_total_load = df_total_load.iloc[:-1]
    df_total_load['# timestamp'] = pd.to_datetime(df_total_load['# timestamp'])
    df_total_load.set_index('# timestamp',inplace=True)
    df_total_load = df_total_load.loc[start:end]
    total_load = (df_total_load.sum(axis=1)/12.).sum() #kWh

    df_WS['res_load'] = df_total_load.sum(axis=1)
    supply_wolosses = (df_WS['res_load']/1000./12.).sum() # only residential load, not what is measured at trafo
    df_WS['res_cost'] = df_WS['res_load']/1000.*df_WS['RT']/12.
    supply_cost_wolosses = df_WS['res_cost'].sum()

    try:
        df_inv_load = pd.read_csv(folder+'/total_P_Out.csv',skiprows=range(8)) #in W
        df_inv_load['# timestamp'] = df_inv_load['# timestamp'].map(lambda x: str(x)[:-4])
        df_inv_load = df_inv_load.iloc[:-1]
        df_inv_load['# timestamp'] = pd.to_datetime(df_inv_load['# timestamp'])
        df_inv_load.set_index('# timestamp',inplace=True)  
        df_inv_load = df_inv_load.loc[start:end]
        PV_supply = (df_inv_load.sum(axis=1)/1000./12.).sum() #in kWh
    except:
        PV_supply = 0.0

    net_demand  = total_load - PV_supply
    retail_kWh = supply_cost_wlosses/net_demand
    retail_kWh_wolosses = supply_cost_wolosses/net_demand

    return retail_kWh, retail_kWh_wolosses


def get_RRloss(run,settings_file,city,market_file):
    ind_base = int(settings_file.split('_')[-2])
    start = pd.to_datetime(settings_file.split('_')[-4])
    end = pd.to_datetime(settings_file.split('_')[-3])
    folder = 'Diss/Diss_' + "{:04d}".format(ind_base)

    retail_kWh, retail_kWh_wolosses = get_retailrate(folder,start,end,city,market_file)
    return (retail_kWh - retail_kWh_wolosses)


def write_global(series_settings,ind,ip_address):
    global_file = 'HH_global.py'
    glm = open(global_file,'w') 
    #import pdb; pdb.set_trace()

    # Flexible houses
    glm.write('import os\n\n')

    glm.write('#Result file\n')
    glm.write('results_folder = \''+series_settings['run']+'/'+series_settings['run']+'_'+"{:04d}".format(ind)+'\'\n')
    glm.write('if not os.path.exists(results_folder):\n')
    glm.write('\tos.makedirs(results_folder)\n\n')

    glm.write('#glm parameters\n')
    glm.write('city = \''+series_settings['city']+'\'\n')
    glm.write('month = \''+series_settings['month']+'\'\n')
    glm.write('start_time_str = \''+series_settings['start_time']+'\'\n')
    glm.write('end_time_str = \''+series_settings['end_time']+'\'\n')
    glm.write('player_dir = \''+series_settings['player_dir']+'\'\n')
    glm.write('tmy_file = \''+series_settings['tmy']+'\'\n')
    glm.write('slack_node = \''+series_settings['slack_node']+'\'\n\n')

    glm.write('#Flexible appliances\n')
    glm.write('settings_file = \'' +str(series_settings['settings_file'])+'\'\n')
    glm.write('flexible_houses = '+str(series_settings['flexible_houses'])+'\n')
    glm.write('PV_share = '+str(float(series_settings['PV_share']))+'\n')
    glm.write('EV_share = '+str(float(series_settings['EV_share']))+'\n')
    glm.write('EV_data = \''+series_settings['EV_data']+'\'\n')
    glm.write('EV_speed = \''+series_settings['EV_speed']+'\'\n')
    glm.write('Batt_share = '+str(float(series_settings['Batt_share']))+'\n')
    glm.write('assert PV_share >= Batt_share, \'More batteries than PV\'\n')
    #glm.write('assert PV_share >= EV_share, \'More EVs than PV\'\n\n')
    
    glm.write('#Market parameters\n')
    glm.write('C = '+str(float(series_settings['line_capacity']))+'\n')
    glm.write('market_data = \''+series_settings['market_data']+'\'\n')
    RR_loss = get_RRloss(series_settings['run'],series_settings['settings_file'],series_settings['city'],series_settings['market_data'])
    glm.write('RR_loss = '+str(RR_loss)+'\n')
    glm.write('p_max = '+str(float(series_settings['p_max']))+'\n')
    glm.write('load_forecast = \''+series_settings['load_forecast']+'\'\n')
    glm.write('unresp_factor = '+str(float(series_settings['unresp_factor']))+'\n')
    glm.write('FIXED_TARIFF = '+str(series_settings['fixed_tariff'])+'\n')
    glm.write('interval = '+str(int(series_settings['interval']))+'\n')
    glm.write('allocation_rule = \''+series_settings['allocation_rule']+'\'\n\n')

    glm.write('#Appliance specifications\n')
    glm.write('delta = '+str(float(series_settings['delta']))+' #temperature bandwidth - HVAC inactivity\n')
    glm.write('ref_price = \''+series_settings['ref_price']+'\'\n')
    glm.write('price_intervals = '+str(int(series_settings['price_intervals']))+' #p average calculation \n')
    glm.write('which_price = \''+series_settings['which_price']+'\' #battery scheduling\n\n')
 
    glm.write('#include System Operator\n')
    glm.write('include_SO = '+str(series_settings['include_SO'])+'\n\n')

    glm.write('#precision in bidding and clearing price\n')
    glm.write('prec = '+str(int(series_settings['prec']))+'\n')
    glm.write('M = '+str(int(series_settings['M']))+' #large number\n')
    glm.write('ip_address = \''+ip_address+'\'\n')

    glm.close()
    return