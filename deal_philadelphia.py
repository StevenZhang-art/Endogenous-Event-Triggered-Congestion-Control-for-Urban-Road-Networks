import pandas as pd
import os

NETWORK_NAME = "Philadelphia"

NODE_CSV = f"{NETWORK_NAME}_node.csv"
NET_CSV = f"{NETWORK_NAME}_net.csv"
OD_CSV = f"{NETWORK_NAME}_od.csv"

OUTPUT_NOD = f"{NETWORK_NAME}.nod.xml"
OUTPUT_EDG = f"{NETWORK_NAME}.edg.xml"
OUTPUT_ODM = f"{NETWORK_NAME}.odm"
OUTPUT_EDGE_CAPACITY = f"edge_capacity_{NETWORK_NAME.lower()}.csv"

MILE_TO_METER = 1609.34
MINUTE_TO_SECOND = 60

print("正在生成节点文件...")
node_df = pd.read_csv(NODE_CSV)

with open(OUTPUT_NOD, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f.write('       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">\n\n')

    for _, row in node_df.iterrows():
        node_id = int(row["Node"])
        x = row["X"]
        y = row["Y"]
        f.write(f'    <node id="{node_id}" x="{x}" y="{y}" type="traffic_light"/>\n')

    f.write('\n</nodes>\n')
print(f"✅ 节点文件生成完成：{OUTPUT_NOD}，共{len(node_df)}个节点")

print("\n正在生成路段文件...")
net_df = pd.read_csv(NET_CSV)

edge_capacity_dict = {}

with open(OUTPUT_EDG, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f.write('       xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">\n\n')

    for idx, row in net_df.iterrows():
        source = int(row["source"])
        target = int(row["target"])
        edge_id = f"{source}_{target}"

        length_m = row["length"] * MILE_TO_METER
        time_seconds = row["freeFlowTime"] * MINUTE_TO_SECOND
        if time_seconds <= 0:
            time_seconds = 1e-6  # 避免除零
        free_flow_speed = min(length_m / time_seconds, 20.0)  # 限制最大速度

        capacity = row["capacity"]
        edge_capacity_dict[edge_id] = capacity

        f.write(f'    <edge id="{edge_id}" from="{source}" to="{target}" '
                f'numLanes="2" speed="{free_flow_speed:.4f}" length="{length_m:.2f}"/>\n')

    f.write('\n</edges>\n')
print(f"✅ 路段文件生成完成：{OUTPUT_EDG}，共{len(net_df)}条路段")

pd.DataFrame.from_dict(
    edge_capacity_dict, orient="index", columns=["capacity_veh_per_hour"]
).to_csv(OUTPUT_EDGE_CAPACITY)
print(f"✅ 路段容量表已保存：{OUTPUT_EDGE_CAPACITY}，可直接导入您的内生算法")

print("\n正在生成OD矩阵文件...")
od_df = pd.read_csv(OD_CSV)

with open(OUTPUT_ODM, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<odMatrix xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n')
    f.write('           xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/odmatrix_file.xsd">\n\n')
    f.write('    <interval begin="0" end="3600">\n')

    for _, row in od_df.iterrows():
        origin = int(row["O"])
        destination = int(row["D"])
        flow = row["Ton"]
        f.write(f'        <odPair origin="{origin}" destination="{destination}" count="{flow}"/>\n')

    f.write('    </interval>\n')
    f.write('</odMatrix>\n')
print(f"✅ OD矩阵文件生成完成：{OUTPUT_ODM}，共{len(od_df)}个OD对")

print("\n🎉 所有文件批量生成完成！")