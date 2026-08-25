import pandas as pd
import numpy as np

NETWORK_NAME = "Philadelphia"

OD_CSV = f"{NETWORK_NAME}_od.csv"
OUTPUT_TRIPS = f"{NETWORK_NAME}.trips.xml"
SIM_DURATION = 3600
SEED = 42
FLOW_SCALE_FACTOR = 1

np.random.seed(SEED)
od_df = pd.read_csv(OD_CSV)

with open(OUTPUT_TRIPS, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f.write('        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n\n')

    veh_counter = 0
    for _, row in od_df.iterrows():
        origin_node = int(row["O"])
        dest_node = int(row["D"])
        flow = int(row["Ton"])

        flow = flow * FLOW_SCALE_FACTOR

        if flow <= 0:
            continue

        interval = SIM_DURATION / flow
        depart_times = np.arange(0, SIM_DURATION, interval)
        depart_times += np.random.uniform(-0.5, 0.5, size=len(depart_times))
        depart_times = np.clip(depart_times, 0.1, SIM_DURATION - 0.1)
        depart_times.sort()

        for depart_time in depart_times:
            veh_counter += 1
            f.write(f'    <trip id="veh_{veh_counter}" depart="{depart_time:.2f}" '
                    f'fromJunction="{origin_node}" toJunction="{dest_node}"/>\n')

    f.write('\n</routes>\n')

print(f"✅ 行程文件生成完成：{OUTPUT_TRIPS}")
print(f"   流量放大系数：{FLOW_SCALE_FACTOR}倍")
print(f"   共生成 {veh_counter} 辆车，覆盖全部OD对")
