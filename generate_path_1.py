import xml.etree.ElementTree as ET

NETWORK_NAME = "Philadelphia"

NET_FILE = f"{NETWORK_NAME}.net.xml"
INPUT_TRIP = f"{NETWORK_NAME}.trips.xml"
OUTPUT_TRIP = f"{NETWORK_NAME}_sorted.trips.xml"
MAX_DEPART_TIME = 3600

junc_first_out_edge = dict()
net_tree = ET.parse(NET_FILE)
net_root = net_tree.getroot()

for edge in net_root.findall("edge"):
    eid = edge.get("id")
    from_junc = edge.get("from")
    if from_junc not in junc_first_out_edge:
        junc_first_out_edge[from_junc] = eid

trip_tree = ET.parse(INPUT_TRIP)
trip_root = trip_tree.getroot()

trips = trip_root.findall("trip")
trips = [t for t in trips if float(t.get("depart")) <= MAX_DEPART_TIME]
trips.sort(key=lambda x: float(x.get("depart")))

for trip in trips:
    junc_id = trip.get("fromJunction")
    if junc_id in junc_first_out_edge:
        edge_id = junc_first_out_edge[junc_id]
        trip.set("from", edge_id)

trip_root[:] = trips

trip_tree.write(
    OUTPUT_TRIP,
    encoding="utf-8",
    xml_declaration=True,
    short_empty_elements=True
)

print(f"排序+补全from属性+时间裁剪完成，输出：{OUTPUT_TRIP}")
print(f"共保留 {len(trips)} 条行程（发车时间 0~{MAX_DEPART_TIME}s）")
