import json
import xml.etree.ElementTree as ET
import html

HEADER = """<mxfile host="app.diagrams.net">
  <diagram id="{id}" name="{name}">
    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="2000" pageHeight="1100" background="#FAFAFA" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
"""

FOOTER = """      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""

def esc(s):
    return html.escape(s).replace('\n', '&lt;br&gt;')

class DiagramBuilder:
    def __init__(self, diag_id, name):
        self.xml = HEADER.format(id=diag_id, name=name)
    
    def add_title(self, title, subtitle):
        self.xml += f'''
        <mxCell id="t1" value="{esc(title)}" style="text;html=1;align=center;verticalAlign=middle;fontSize=32;fontStyle=1;fontColor=#111827;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="500" y="30" width="1000" height="40" as="geometry"/></mxCell>
        <mxCell id="t2" value="{esc(subtitle)}" style="text;html=1;align=center;fontSize=14;fontColor=#6B7280;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="500" y="70" width="1000" height="20" as="geometry"/></mxCell>
        '''
        
    def add_zone(self, z_id, title, x, y, w, h, border_color="#E5E7EB", dashed=True):
        dash_str = "dashed=1;dashPattern=5 5;" if dashed else ""
        self.xml += f'''
        <mxCell id="{z_id}" value="{esc(title)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=none;strokeColor={border_color};strokeWidth=2;{dash_str}arcSize=2;verticalAlign=top;align=left;spacingTop=10;spacingLeft=15;fontStyle=1;fontSize=12;fontColor={border_color};" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>
        '''
        
    def add_node(self, n_id, html_value, x, y, w, h, fill="#ffffff", stroke="#d1d5db", font_color="#111827", is_dashed=False):
        dash = "dashed=1;" if is_dashed else ""
        self.xml += f'''
        <mxCell id="{n_id}" value="{esc(html_value)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.5;fontColor={font_color};{dash}shadow=0;arcSize=6;fontSize=12;" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>
        '''

    def add_edge(self, e_id, src, dst, label="", color="#9CA3AF", edge_style="orthogonalEdgeStyle;",
                  is_dashed=False, exit_x=None, exit_y=None, entry_x=None, entry_y=None, waypoints=None):
        dash = "dashed=1;" if is_dashed else ""
        ex = f'exitX="{exit_x}" exitY="{exit_y}" exitDx="0" exitDy="0" ' if exit_x is not None else ""
        en = f'entryX="{entry_x}" entryY="{entry_y}" entryDx="0" entryDy="0" ' if entry_x is not None else ""
        edge_style_full = f"edgeStyle={edge_style};rounded=1;strokeColor={color};strokeWidth=1.5;html=1;fontColor={color};fontSize=11;{dash}"
        pts = ""
        if waypoints:
            inner = "".join(f'<mxPoint x="{wx}" y="{wy}"/>' for wx, wy in waypoints)
            pts = f'<Array as="points">{inner}</Array>'
        self.xml += f'''
        <mxCell id="{e_id}" value="{esc(label)}" style="{edge_style_full}" edge="1" parent="1" source="{src}" target="{dst}" {ex}{en}>
            <mxGeometry relative="1" as="geometry">{pts}</mxGeometry>
        </mxCell>
        '''

    def build(self, filename):
        self.xml += FOOTER
        with open(filename, 'w') as f:
            f.write(self.xml)

# STYLE CONSTANTS
FILL_MOB = "#F0F9FF"; BORD_MOB = "#0284C7"; FONT_MOB = "#0369A1"
FILL_LB = "#F0FDF4"; BORD_LB = "#16A34A"; FONT_LB = "#15803D"
FILL_OR = "#EFF6FF"; BORD_OR = "#2563EB"; FONT_OR = "#1D4ED8"
FILL_QSV = "#FFF7ED"; BORD_QSV = "#EA580C"; FONT_QSV = "#C2410C"
FILL_EDG = "#FAF5FF"; BORD_EDG = "#9333EA"; FONT_EDG = "#7E22CE"
FILL_CDN = "#F0FDFA"; BORD_CDN = "#0D9488"; FONT_CDN = "#0F766E"
FILL_VW = "#FEF2F2"; BORD_VW = "#DC2626"; FONT_VW = "#B91C1C"
Y_BASE = 150
ZONE_H = 480

def generate_global_arch():
    d = DiagramBuilder("global_arch", "Global Architecture")
    d.add_title("Live Streaming Component Diagram", "AWS/C4 Style Representation of System Flow and Boundaries")
    
    d.add_zone("z_1", "1. SOURCE TIER", 50, Y_BASE, 200, ZONE_H, "#64748B")
    d.add_zone("z_2", "2. INGESTION GATEWAY", 290, Y_BASE, 200, ZONE_H, BORD_LB)
    d.add_zone("z_3", "3. ORIGIN STORAGE", 530, Y_BASE, 200, ZONE_H, BORD_OR)
    d.add_zone("z_4", "4. TRANSCODING FARM", 770, Y_BASE, 220, ZONE_H, BORD_QSV)
    d.add_zone("z_5", "5. EDGE PACKAGER", 1030, Y_BASE, 200, ZONE_H, BORD_EDG)
    d.add_zone("z_6", "6. GLOBAL CDN", 1270, Y_BASE, 200, ZONE_H, BORD_CDN)
    d.add_zone("z_7", "7. CONSUMER EDGE", 1510, Y_BASE, 200, ZONE_H, BORD_VW)

    d.add_node("app1", "iOS Device\n(HaishinKit)", 70, 200, 160, 60, FILL_MOB, BORD_MOB, FONT_MOB)
    d.add_node("app2", "Android Device\n(HaishinKit)", 70, 280, 160, 60, FILL_MOB, BORD_MOB, FONT_MOB)

    d.add_node("hap1", "HAProxy LB (Primary)\nTCP Load Balancing", 310, 240, 160, 60, FILL_LB, BORD_LB, FONT_LB)
    d.add_node("hap2", "HAProxy LB (Failover)\nKeepAlived VIP", 310, 340, 160, 60, FILL_LB, BORD_LB, FONT_LB, True)

    d.add_node("org1", "SRS Origin Server", 550, 220, 160, 60, FILL_OR, BORD_OR, FONT_OR)
    d.add_node("redis", "Redis CPX11\n(Auth & Metadata)", 550, 310, 160, 50, FILL_OR, BORD_OR, FONT_OR)

    d.add_node("qsv_lbl", "Intel QSV Cluster", 790, 180, 180, 40, FILL_QSV, BORD_QSV, FONT_QSV, True)
    d.add_node("ex1", "Intel QSV Node 1\n(i5-13500 UHD 770)", 790, 240, 180, 60, FILL_QSV, BORD_QSV, FONT_QSV)
    d.add_node("ex2", "Intel QSV Node N\n(Auto-Scaled)", 790, 330, 180, 60, FILL_QSV, BORD_QSV, FONT_QSV, True)

    d.add_node("edg", "SRS Edge Server\n(HTTP-FLV Output)", 1050, 260, 160, 80, FILL_EDG, BORD_EDG, FONT_EDG)

    d.add_node("cdn1", "Bunny CDN", 1290, 200, 160, 60, FILL_CDN, BORD_CDN, FONT_CDN)
    d.add_node("cdn2", "Gcore CDN", 1290, 300, 160, 60, FILL_CDN, BORD_CDN, FONT_CDN)

    d.add_node("view", "Client Players\nAdaptive Bitrate\n(360p - 720p)", 1530, 250, 160, 80, FILL_VW, BORD_VW, FONT_VW)

    arr = "orthogonalEdgeStyle;"
    b = "#9CA3AF"
    # Source -> Ingestion Gateway
    d.add_edge("a1", "app1", "hap1", "RTMP/SRT", b, arr)
    d.add_edge("a2", "app2", "hap1", "", b, arr)
    # hap1 -> hap2 Failover: route LEFT of hap1 then down to avoid crossing
    d.add_edge("a2b", "hap1", "hap2", "Failover", b, arr, is_dashed=True,
               exit_x=0, exit_y=0.5, entry_x=0, entry_y=0.5,
               waypoints=[(285, 370)])
    # Ingestion -> Origin
    d.add_edge("a3", "hap1", "org1", "Ingest (vRack)", b, arr)
    # Origin -> Redis: route RIGHT of zone to avoid crossing over other nodes
    d.add_edge("a3b", "org1", "redis", "Token Auth", b, arr,
               exit_x=1, exit_y=0.5, entry_x=1, entry_y=0.5,
               waypoints=[(760, 335)])
    # Origin -> Transcoding (Lazy Pull)
    d.add_edge("a4", "org1", "ex1", "Lazy Pull", b, arr)
    # Transcoding -> Edge Packager
    d.add_edge("a5", "ex1", "edg", "x4 ABR Streams", b, arr)
    # Edge -> CDN (both)
    d.add_edge("a6", "edg", "cdn1", "HTTP-FLV", b, arr)
    d.add_edge("a7", "edg", "cdn2", "HTTP-FLV", b, arr)
    # CDN -> Viewers (both)
    d.add_edge("a8", "cdn1", "view", "Delivery", b, arr)
    d.add_edge("a9", "cdn2", "view", "", b, arr)

    d.build("diagrams/shopfeed-live-infra.drawio")

def generate_phase(name, title, desc, config):
    d = DiagramBuilder(name, title)
    d.add_title(title, desc)
    d.xml = d.xml.replace('pageWidth="2000"', 'pageWidth="2300"')
    
    d.add_zone("z_mob", "INGESTION TIER", 40, Y_BASE, 180, ZONE_H, "#94A3B8")
    d.add_zone("z_lb", "LOAD BALANCING TIER", 260, Y_BASE, 200, ZONE_H, BORD_LB)
    d.add_zone("z_or", "ORIGIN & CONTROL TIER", 500, Y_BASE, 240, ZONE_H, BORD_OR)
    d.add_zone("z_qsv", "TRANSCODING TIER (Intel QSV)", 780, Y_BASE, 300, ZONE_H, BORD_QSV)
    d.add_zone("z_edg", "DISTRIBUTION EDGE", 1120, Y_BASE, 200, ZONE_H, BORD_EDG)
    d.add_zone("z_cdn", "CDN LAYER", 1360, Y_BASE, 200, ZONE_H, BORD_CDN)
    d.add_zone("z_vw", "CONSUMPTION TIER", 1600, Y_BASE, 200, ZONE_H, BORD_VW)

    # Base common nodes
    d.add_node("ios", "iOS SDK\nHaishinKit (RTMP)", 60, 200, 140, 60, FILL_MOB, BORD_MOB, FONT_MOB)
    d.add_node("and", "Android SDK\nHaishinKit (RTMP)", 60, 280, 140, 60, FILL_MOB, BORD_MOB, FONT_MOB)

    d.add_node("vip", "Keepalived VIP\n(Failover < 2s)", 280, 240, 160, 40, FILL_LB, BORD_LB, FONT_LB)
    d.add_node("lb_mas", "HAProxy Master\nCX33", 280, 300, 160, 60, FILL_LB, BORD_LB, FONT_LB)
    d.add_node("lb_bck", "HAProxy Backup\nCX33", 280, 400, 160, 60, FILL_LB, BORD_LB, FONT_LB, True)

    d.add_node("ori_1", config["ori"], 520, 220, 200, 60, FILL_OR, BORD_OR, FONT_OR)
    if "ori_ha" in config:
        d.add_node("ori_2", config["ori_ha"], 520, 300, 200, 60, FILL_OR, BORD_OR, FONT_OR, True)
    
    d.add_node("redis", config["redis"], 520, 400, 200, 50, FILL_OR, BORD_OR, FONT_OR)
    
    # QSV nodes depending on phase
    d.add_node("qsv_cap", config["qsv_cap"], 800, 180, 260, 50, FILL_QSV, BORD_QSV, FONT_QSV)
    d.add_node("qsv_act", config["qsv_act"], 800, 250, 260, 90, FILL_QSV, BORD_QSV, FONT_QSV)
    d.add_node("qsv_sb", config["qsv_sb"], 800, 360, 260, 60, FILL_QSV, BORD_QSV, FONT_QSV, True)

    # Edge nodes
    d.add_node("edg_1", config["edg_1"], 1140, 220, 160, 60, FILL_EDG, BORD_EDG, FONT_EDG)
    d.add_node("edg_2", config["edg_2"], 1140, 320, 160, 60, FILL_EDG, BORD_EDG, FONT_EDG, "is_dashed" in config and config.get("is_dashed"))

    d.add_node("cdn_bun", "Bunny CDN Premium", 1380, 180, 160, 60, FILL_CDN, BORD_CDN, FONT_CDN)
    d.add_node("cdn_gco", "Gcore CDN Default", 1380, 280, 160, 60, FILL_CDN, BORD_CDN, FONT_CDN)
    if "cdn_p2p" in config:
        d.add_node("cdn_p2p", config["cdn_p2p"], 1380, 380, 160, 50, FILL_CDN, BORD_CDN, FONT_CDN, True)

    d.add_node("vw", config["vw"], 1620, 260, 160, 100, FILL_VW, BORD_VW, FONT_VW)
    
    d.add_node("mon", config["mon"], 40, 650, 1760, 40, "#1F2937", "#111827", "#F9FAFB")

    # EDGES — Full verified flow with waypoints to avoid crossing
    e = "orthogonalEdgeStyle"
    b = "#9CA3AF"
    # Mobile apps -> VIP
    d.add_edge("e1", "ios", "vip", "RTMP", color=b, edge_style=e)
    d.add_edge("e2", "and", "vip", "RTMP", color=b, edge_style=e)
    # VIP -> HAProxy Master (straight down, Active)
    d.add_edge("e3", "vip", "lb_mas", "Active", color=b, edge_style=e,
               exit_x=0.5, exit_y=1, entry_x=0.5, entry_y=0)
    # VIP -> HAProxy Backup: route LEFT side to bypass lb_mas
    d.add_edge("e4", "vip", "lb_bck", "Standby", is_dashed=True, color=b, edge_style=e,
               exit_x=0, exit_y=0.5, entry_x=0, entry_y=0,
               waypoints=[(250, 430)])
    # HAProxy Master -> Origin Primary (horizontal)
    d.add_edge("e5", "lb_mas", "ori_1", "Ingest (vRack)", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # Origin Primary -> Redis: route RIGHT side to avoid crossing Origin Standby
    d.add_edge("e5b", "ori_1", "redis", "Token Auth", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=1, entry_y=0.5,
               waypoints=[(770, 425)])
    # Origin -> QSV (Lazy Pull)
    d.add_edge("e6", "ori_1", "qsv_act", "Lazy Pull", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # QSV -> Edge Node 1
    d.add_edge("e7", "qsv_act", "edg_1", "ABR Output", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # QSV -> Edge Node 2
    d.add_edge("e7b", "qsv_act", "edg_2", "", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # Edge 1 -> Bunny CDN
    d.add_edge("e8", "edg_1", "cdn_bun", "HTTP-FLV", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # Edge 2 -> Gcore CDN
    d.add_edge("e8b", "edg_2", "cdn_gco", "HTTP-FLV", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # Both CDNs -> Viewers
    d.add_edge("e9", "cdn_bun", "vw", "Stream", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    d.add_edge("e9b", "cdn_gco", "vw", "", color=b, edge_style=e,
               exit_x=1, exit_y=0.5, entry_x=0, entry_y=0.5)
    # P2P CDN -> Viewers (if present)
    if "cdn_p2p" in config:
        d.add_edge("e9c", "cdn_p2p", "vw", "P2P", color=b, edge_style=e, is_dashed=True,
                   exit_x=1, exit_y=0.5, entry_x=0, entry_y=1)

    d.build(f"diagrams/{name}.drawio")

def generate_roadmap():
    d = DiagramBuilder("roadmap", "Evolution Roadmap")
    d.xml = d.xml.replace('pageWidth="2000"', 'pageWidth="2200"')
    d.add_title("Capacity Scaling Lifecycle", "Predictable infrastructure evolution from MVP to Enterprise Scale")
    
    Y_BASE = 150
    ZONE_H = 650
    W = 400
    
    d.add_zone("p1a", "PHASE 1A: LAUNCH\n45 Concurrent Lives", 100, Y_BASE, W, ZONE_H, "#16A34A", False)
    d.add_zone("p1", "PHASE 1: MVP\n300 Concurrent Lives", 550, Y_BASE, W, ZONE_H, "#2563EB", False)
    d.add_zone("p2", "PHASE 2: SCALE\n1 000 Concurrent Lives", 1000, Y_BASE, W, ZONE_H, "#EA580C", False)
    d.add_zone("p3", "PHASE 3: PRODUCTION\n2 000+ Concurrent Lives", 1450, Y_BASE, W, ZONE_H, "#DC2626", False)

    phases_data = [
        ("p1a", 100, [
            ("Target", "36 active streams (80% ratio)"),
            ("Ingest", "2x CX33 (Keepalived HA)"),
            ("Origin", "1x Rise-1"),
            ("QSV", "3 Active + 1 Standby (EX44)"),
            ("Edge", "1x Rise-2 (1Gbps)"),
            ("Database", "1x CPX11 Redis"),
            ("Observability", "1x CX33 Prometheus/Grafana"),
            ("Estimated OPEX", "EUR 370 ~ 560 / month")
        ], "#F0FDF4", "#15803D"),
        ("p1", 550, [
            ("Target", "240 active streams (80% ratio)"),
            ("Ingest", "2x CX33"),
            ("Origin", "1x Rise-1"),
            ("QSV", "20 Active + 1 Standby (EX44)"),
            ("Edge", "3x Rise-2 (1Gbps)"),
            ("Database", "1x CPX11 Redis"),
            ("Observability", "1x AX41 NVMe HA"),
            ("Estimated OPEX", "EUR 1 622 ~ 2 722 / month")
        ], "#EFF6FF", "#1D4ED8"),
        ("p2", 1000, [
            ("Target", "800 active streams (80% ratio)"),
            ("Ingest", "2x CX33"),
            ("Origin", "2x Rise-1 (Primary/Secondary)"),
            ("QSV", "50+ EX44 (Cluster Auto-scale)"),
            ("Edge", "3x Rise-2 (2Gbps limit)"),
            ("Database", "2x CPX32 Redis Sentinel"),
            ("Observability", "1x AX41 NVMe + Alerts"),
            ("Estimated OPEX", "EUR 3 385 ~ 5 085 / month")
        ], "#FFF7ED", "#C2410C"),
        ("p3", 1450, [
            ("Target", "1 600 active streams (80% ratio)"),
            ("Ingest", "2x CX33"),
            ("Origin", "2x Rise-1"),
            ("QSV", "88 to 117 EX44 (Virtual Racks)"),
            ("Edge", "4+ Rise-2 (2Gbps limit)"),
            ("Database", "2x CPX32 Redis Sentinel"),
            ("Observability", "2x AX41 NVMe + Auto-Order"),
            ("Estimated OPEX", "EUR 5 946 ~ 11 567 / month")
        ], "#FEF2F2", "#B91C1C")
    ]

    for p_id, x, items, bg_col, text_col in phases_data:
        y_curr = 220
        for label, val in items:
            d.add_node(f"{p_id}_{y_curr}_lbl", label, x+20, y_curr, 120, 40, "none", "none", "#4B5563")
            d.add_node(f"{p_id}_{y_curr}_val", val, x+150, y_curr, 230, 40, bg_col, text_col, text_col)
            y_curr += 60
            
    d.add_edge("tr1", "p1a", "p1", "Scale Traffic", edge_style="orthogonalEdgeStyle")
    d.add_edge("tr2", "p1", "p2", "Scale Engine", edge_style="orthogonalEdgeStyle")
    d.add_edge("tr3", "p2", "p3", "Enterprise HA", edge_style="orthogonalEdgeStyle")

    d.build("diagrams/scaling-roadmap.drawio")

# Execute scripts
generate_global_arch()
generate_roadmap()

generate_phase("phase1a", "Phase 1A: Quick Launch", "Initial Validation Architecture (45 Lives)", {
    "ori": "SRS Origin Primary\nRise-1",
    "redis": "Redis Metadata\nCPX11",
    "qsv_cap": "Transcoding Capacity\n45 Active Streams",
    "qsv_act": "3x EX44 (i5-13500)\nActive Transcoders",
    "qsv_sb": "1x EX44\nHot Standby Pool",
    "edg_1": "SRS Edge Node\nRise-2 (1Gbps)",
    "edg_2": "Scale Slot",
    "is_dashed": True,
    "vw": "Trial Viewers\nHTTP-FLV < 3s",
    "mon": "OBSERVABILITY SUBSYSTEM: CX33 (Prometheus, Grafana, Lightweight Logging)"
})

generate_phase("phase1-mvp", "Phase 1: MVP Scale", "Product Validation Architecture (300 Lives)", {
    "ori": "SRS Origin Primary\nRise-1",
    "redis": "Redis Metadata\nCPX11",
    "qsv_cap": "Transcoding Capacity\n300 Active Streams",
    "qsv_act": "20x EX44 (i5-13500)\nActive Transcoders",
    "qsv_sb": "1x EX44\nHot Standby Pool",
    "edg_1": "SRS Edge Node 1\nRise-2 (1Gbps)",
    "edg_2": "SRS Edge Nodes 2,3\nRise-2 (1Gbps)",
    "is_dashed": False,
    "vw": "Beta Viewers\nHTTP-FLV < 3s\nABR 4 qualities",
    "mon": "OBSERVABILITY SUBSYSTEM: AX41 NVMe (Prometheus, Grafana, Full Alerting)"
})

generate_phase("phase2-scale", "Phase 2: High Scale", "Growth Architecture (1 000 Lives)", {
    "ori": "SRS Origin Primary\nRise-1",
    "ori_ha": "SRS Origin Standby\nRise-1 (Keepalived)",
    "redis": "Redis Sentinel HA\n2x CPX32",
    "qsv_cap": "Transcoding Capacity\n800 Active Streams",
    "qsv_act": "50+ EX44 Cluster\nHorizontally Scaled",
    "qsv_sb": "5x EX44\nHot Standby Pool",
    "edg_1": "SRS Edge Node 1 (Core)\nRise-2 (2Gbps vRack)",
    "edg_2": "SRS Edge Nodes 2,3\nRise-2 (2Gbps vRack)",
    "is_dashed": False,
    "vw": "Full User Base\nScale CDN Delivery",
    "mon": "OBSERVABILITY HA: AX41 NVMe + Slack Webhooks + Scaling Advisor Rules"
})

generate_phase("phase3-production", "Phase 3: Production Master", "Enterprise High Availability Architecture (2 000+ Lives)", {
    "ori": "SRS Origin Primary\nRise-1",
    "ori_ha": "SRS Origin Standby\nRise-1 (Keepalived)",
    "redis": "Redis Sentinel HA\n2x CPX32",
    "qsv_cap": "Transcoding Capacity\n1 600 - 1 755 Active Streams",
    "qsv_act": "Logical Rack A, B, C, D\n88 to 117 EX44 Servers\n(API Ordered)",
    "qsv_sb": "10x EX44\nHot Standby Pool",
    "edg_1": "SRS Edge Nodes 1, 2\nRise-2 (2Gbps vRack)",
    "edg_2": "SRS Edge Nodes 3, 4+\nRise-2 (2Gbps vRack)",
    "is_dashed": False,
    "cdn_p2p": "Streamroot P2P Mesh\nPeer-to-Peer Transfer",
    "vw": "Unlimited Viewers\nMulti-CDN Offload\nP2P Data saving",
    "mon": "OBSERVABILITY HA: 2x AX41 NVMe + Automated Hardware Provisioning Engine"
})

print("All modern AWS/C4 Style diagrams generated flawlessly!")
