import fixed_order

first_hour = 4000
timesteps = 100
fixed_order.FixedOrder('commonen.xlsx', 'hp_0_ts_15000st_1500').run_timesteps(
    first_hour, timesteps)

