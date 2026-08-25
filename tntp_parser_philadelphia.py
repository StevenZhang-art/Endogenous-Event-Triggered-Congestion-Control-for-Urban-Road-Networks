import os
import re
import pandas as pd

NETWORK_NAME = "Philadelphia"

NODE_TNTP = "Philadelphia_node.tntp"
NET_TNTP = "Philadelphia_net.tntp"
TRIPS_TNTP = "Philadelphia_trips.tntp"

NODE_CSV = f"{NETWORK_NAME}_node.csv"
NET_CSV = f"{NETWORK_NAME}_net.csv"
OD_CSV = f"{NETWORK_NAME}_od.csv"


def parse_node_tntp(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            rows.append((node_id, x, y))
    return pd.DataFrame(rows, columns=["Node", "X", "Y"])


def parse_net_tntp(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        past_metadata = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not past_metadata:
                if stripped.startswith("<END OF METADATA>"):
                    past_metadata = True
                continue
            if stripped.startswith("~"):
                continue
            content = stripped.split(";")[0].strip()
            if not content:
                continue
            parts = content.split()
            init_node = int(parts[0])
            term_node = int(parts[1])
            capacity = float(parts[2])
            length = float(parts[3])
            free_flow_time = float(parts[4])
            rows.append((init_node, term_node, length, free_flow_time, capacity))
    return pd.DataFrame(rows, columns=["source", "target", "length", "freeFlowTime", "capacity"])


def parse_trips_tntp(path):
    origin_pattern = re.compile(r"^Origin\s+(\d+)")
    pair_pattern = re.compile(r"(\d+)\s*:\s*([\d.eE+-]+)")

    rows = []
    current_origin = None
    current_block_lines = []

    def flush_block(origin, block_lines):
        if origin is None:
            return
        block_text = " ".join(block_lines)
        for dest_str, flow_str in pair_pattern.findall(block_text):
            flow = float(flow_str)
            if flow == 0:
                continue
            rows.append((origin, int(dest_str), flow))

    with open(path, "r", encoding="utf-8") as f:
        past_metadata = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if not past_metadata:
                if stripped.startswith("<END OF METADATA>"):
                    past_metadata = True
                continue
            origin_match = origin_pattern.match(stripped)
            if origin_match:
                flush_block(current_origin, current_block_lines)
                current_origin = int(origin_match.group(1))
                current_block_lines = []
                continue
            current_block_lines.append(stripped)

    flush_block(current_origin, current_block_lines)

    return pd.DataFrame(rows, columns=["O", "D", "Ton"])


def run(source_dir, output_dir):
    node_df = parse_node_tntp(os.path.join(source_dir, NODE_TNTP))
    net_df = parse_net_tntp(os.path.join(source_dir, NET_TNTP))
    od_df = parse_trips_tntp(os.path.join(source_dir, TRIPS_TNTP))

    node_out = os.path.join(output_dir, NODE_CSV)
    net_out = os.path.join(output_dir, NET_CSV)
    od_out = os.path.join(output_dir, OD_CSV)

    node_df.to_csv(node_out, index=False)
    net_df.to_csv(net_out, index=False)
    od_df.to_csv(od_out, index=False)

    print(f"Parsed {len(node_df)} nodes -> {node_out}")
    print(f"Parsed {len(net_df)} links -> {net_out}")
    print(f"Parsed {len(od_df)} nonzero OD pairs -> {od_out}")


def main():
    run(source_dir=os.getcwd(), output_dir=os.getcwd())


if __name__ == "__main__":
    main()
