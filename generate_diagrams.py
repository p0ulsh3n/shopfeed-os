import os

HEADER = '''<mxfile host="app.diagrams.net">
  <diagram id="{diag_id}" name="{diag_name}">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="900" background="#FFFFFF" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
'''

FOOTER = '''      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

def esc(text):
    return text.replace('\n', '<br>')

def create_global_arch():
    xml = HEADER.format(diag_id="global-arch", diag_name="Architecture Globale")
    
    # Title
    xml += f'''
        <mxCell id="title" value="ShopFeed Live Streaming - Architecture Globale" style="text;html=1;align=center;verticalAlign=middle;fontSize=28;fontStyle=1;fontColor=#1E293B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="30" width="1000" height="40" as="geometry"/></mxCell>
        <mxCell id="subtitle" value="Pipeline RTMP/SRT -> Intel QSV -> Multi-CDN -> Viewers Worldwide" style="text;html=1;align=center;fontSize=14;fontColor=#64748B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="70" width="1000" height="20" as="geometry"/></mxCell>
    '''
    
    zones = [
        ("Mobile", 40, "#F0F9FF", "#0284C7"),
        ("Ingest HA", 260, "#F0FDF4", "#16A34A"),
        ("Origin", 480, "#EFF6FF", "#2563EB"),
        ("Intel QSV", 700, "#FFF7ED", "#EA580C"),
        ("Edge", 920, "#FAF5FF", "#9333EA"),
        ("Multi-CDN", 1140, "#F0FDFA", "#0D9488"),
        ("Viewers", 1360, "#FEF2F2", "#DC2626")
    ]
    
    for i, (name, x, fill, stroke) in enumerate(zones):
        xml += f'''
        <mxCell id="z{i}" value="" style="rounded=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;dashed=1;arcSize=4;" vertex="1" parent="1"><mxGeometry x="{x}" y="120" width="180" height="380" as="geometry"/></mxCell>
        <mxCell id="zl{i}" value="{name}" style="text;html=1;align=center;fontStyle=1;fontSize=14;fontColor={stroke};strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="{x}" y="130" width="180" height="30" as="geometry"/></mxCell>
        '''
        
    box_style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#1E293B;shadow=1;arcSize=8;fontSize=12;"
    green_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#22C55E;strokeColor=#16A34A;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"
    blue_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#3B82F6;strokeColor=#2563EB;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"
    orange_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#F97316;strokeColor=#EA580C;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"
    purple_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#A855F7;strokeColor=#9333EA;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"
    teal_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#14B8A6;strokeColor=#0D9488;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"
    red_box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#EF4444;strokeColor=#DC2626;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;"

    # 0 Mobile
    xml += f'''<mxCell id="ios" value="iOS App<br>HaishinKit 2.2.5<br>RTMP + SRT" style="{blue_box}" vertex="1" parent="1"><mxGeometry x="60" y="180" width="140" height="70" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="and" value="Android App<br>HaishinKit.kt<br>RTMP" style="{blue_box}" vertex="1" parent="1"><mxGeometry x="60" y="280" width="140" height="70" as="geometry"/></mxCell>'''

    # 1 HAProxy
    xml += f'''<mxCell id="ha" value="HAProxy LB<br>2x CX33<br>Keepalived VIP<br>Active/Passive" style="{green_box}" vertex="1" parent="1"><mxGeometry x="280" y="230" width="140" height="90" as="geometry"/></mxCell>'''

    # 2 Origin
    xml += f'''<mxCell id="srs" value="SRS Origin<br>OVH Rise-1<br>6C/64GB/2TB<br>JWT Auth" style="{blue_box}" vertex="1" parent="1"><mxGeometry x="500" y="180" width="140" height="80" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="orc" value="Orchestrateur<br>Flask Docker" style="{box_style}" vertex="1" parent="1"><mxGeometry x="500" y="280" width="140" height="50" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="red" value="Redis CPX11" style="{box_style}" vertex="1" parent="1"><mxGeometry x="500" y="350" width="140" height="50" as="geometry"/></mxCell>'''

    # 3 QSV
    xml += f'''<mxCell id="qsv" value="EX44 Workers<br>i5-13500 UHD 770<br>2x MFX Engines<br>15 streams/server" style="{orange_box}" vertex="1" parent="1"><mxGeometry x="720" y="230" width="140" height="90" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="abr" value="4 profils ABR<br>hd5 720p | zsd5 720p<br>sd5 480p | ld5 360p" style="text;html=1;align=center;fontSize=11;fontColor=#EA580C;fillColor=#FFF7ED;strokeColor=#EA580C;rounded=1;" vertex="1" parent="1"><mxGeometry x="720" y="340" width="140" height="50" as="geometry"/></mxCell>'''

    # 4 Edge
    xml += f'''<mxCell id="edg" value="SRS Edge<br>OVH Rise-2<br>HTTP-FLV output<br>1-2 Gbps public<br>vRack prive" style="{purple_box}" vertex="1" parent="1"><mxGeometry x="940" y="230" width="140" height="100" as="geometry"/></mxCell>'''

    # 5 CDN
    xml += f'''<mxCell id="bun" value="Bunny CDN<br>Europe + USA<br>119 PoPs" style="{teal_box}" vertex="1" parent="1"><mxGeometry x="1160" y="180" width="140" height="70" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="gc" value="Gcore CDN<br>Afrique + Asie<br>210+ PoPs" style="{teal_box}" vertex="1" parent="1"><mxGeometry x="1160" y="280" width="140" height="70" as="geometry"/></mxCell>'''

    # 6 Viewers
    xml += f'''<mxCell id="vw" value="Viewers<br>HTTP-FLV < 3s<br>ABR adaptatif<br>iOS + Android<br>CDN scale auto" style="{red_box}" vertex="1" parent="1"><mxGeometry x="1380" y="230" width="140" height="100" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="vm" value="Metrics:<br>TTFF < 1.5s (4G)<br>60K viewers/live" style="text;html=1;align=center;fontSize=11;fontColor=#DC2626;fillColor=#FEF2F2;strokeColor=#DC2626;rounded=1;" vertex="1" parent="1"><mxGeometry x="1380" y="350" width="140" height="50" as="geometry"/></mxCell>'''

    # Arrows
    arr = "edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor={0};strokeWidth=2;fontColor={0};fontSize=11;html=1;"
    xml += f'''
    <mxCell id="a1" style="{arr.format('#2563EB')}" edge="1" parent="1" source="ios" target="ha"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a1b" style="{arr.format('#2563EB')}dashed=1;" edge="1" parent="1" source="and" target="ha"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a2" value="RTMP" style="{arr.format('#16A34A')}" edge="1" parent="1" source="ha" target="srs"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a3" value="Lazy pull" style="{arr.format('#EA580C')}" edge="1" parent="1" source="srs" target="qsv"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a4" value="4 ABR" style="{arr.format('#9333EA')}" edge="1" parent="1" source="qsv" target="edg"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a5" value="HTTP-FLV" style="{arr.format('#0D9488')}" edge="1" parent="1" source="edg" target="bun"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a5b" style="{arr.format('#0D9488')}dashed=1;" edge="1" parent="1" source="edg" target="gc"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a6" style="{arr.format('#DC2626')}" edge="1" parent="1" source="bun" target="vw"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a6b" style="{arr.format('#DC2626')}dashed=1;" edge="1" parent="1" source="gc" target="vw"><mxGeometry relative="1" as="geometry"/></mxCell>
    '''

    # Monitoring block
    xml += f'''<mxCell id="mon" value="MONITORING LAYER: Prometheus + Grafana + AlertManager + Loki | scaling-advisor.py :9099 | Auto-deploy" style="rounded=1;fillColor=#1E293B;fontColor=#F8FAFC;strokeColor=#0F172A;fontSize=12;fontStyle=1;arcSize=6;" vertex="1" parent="1"><mxGeometry x="40" y="530" width="1500" height="40" as="geometry"/></mxCell>'''

    xml += FOOTER
    with open('diagrams/shopfeed-live-infra.drawio', 'w') as f: f.write(xml)

def create_phase(phase_num, title, subtitle, servers):
    filename = f"diagrams/phase{phase_num}.drawio"
    xml = HEADER.format(diag_id=f"phase{phase_num}", diag_name=title)
    
    xml += f'''
        <mxCell id="title" value="{title}" style="text;html=1;align=center;verticalAlign=middle;fontSize=28;fontStyle=1;fontColor=#1E293B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="30" width="1000" height="40" as="geometry"/></mxCell>
        <mxCell id="subtitle" value="{subtitle}" style="text;html=1;align=center;fontSize=14;fontColor=#64748B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="70" width="1000" height="20" as="geometry"/></mxCell>
    '''

    zones = [
        ("Mobile / Ingest HA", 40, "#F0FDF4", "#16A34A", 240),
        ("Origin & Orchestration", 310, "#EFF6FF", "#2563EB", 240),
        ("Intel QSV Transcoding Grid", 580, "#FFF7ED", "#EA580C", 460),
        ("Edge & Distribution", 1070, "#FAF5FF", "#9333EA", 240),
        ("Viewers", 1340, "#FEF2F2", "#DC2626", 200)
    ]
    
    for i, (name, x, fill, stroke, w) in enumerate(zones):
        xml += f'''
        <mxCell id="z{i}" value="" style="rounded=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;dashed=1;arcSize=4;" vertex="1" parent="1"><mxGeometry x="{x}" y="120" width="{w}" height="420" as="geometry"/></mxCell>
        <mxCell id="zl{i}" value="{name}" style="text;html=1;align=center;fontStyle=1;fontSize=14;fontColor={stroke};strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="{x}" y="130" width="{w}" height="30" as="geometry"/></mxCell>
        '''

    box_style = "rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#1E293B;shadow=1;arcSize=8;fontSize=12;align=center;"
    
    # 0 Mobile/Ingest
    xml += f'''<mxCell id="ios" value="Mobile Apps" style="{box_style}" vertex="1" parent="1"><mxGeometry x="60" y="180" width="200" height="50" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="ha" value="{servers['lb']}" style="{box_style}" vertex="1" parent="1"><mxGeometry x="60" y="260" width="200" height="80" as="geometry"/></mxCell>'''

    # 1 Origin
    xml += f'''<mxCell id="srs" value="{servers['origin']}" style="{box_style}" vertex="1" parent="1"><mxGeometry x="330" y="180" width="200" height="80" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="red" value="{servers['redis']}" style="{box_style}" vertex="1" parent="1"><mxGeometry x="330" y="280" width="200" height="60" as="geometry"/></mxCell>'''

    # 2 QSV Grid
    qsv_text = servers['qsv'].replace('|', '<br>')
    xml += f'''<mxCell id="qsv" value="{qsv_text}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F97316;strokeColor=#EA580C;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=14;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="600" y="180" width="420" height="240" as="geometry"/></mxCell>'''

    # 3 Edge/CDN
    xml += f'''<mxCell id="edg" value="{servers['edge']}" style="{box_style}" vertex="1" parent="1"><mxGeometry x="1090" y="180" width="200" height="120" as="geometry"/></mxCell>'''
    xml += f'''<mxCell id="cdn" value="{servers['cdn']}" style="{box_style}" vertex="1" parent="1"><mxGeometry x="1090" y="320" width="200" height="80" as="geometry"/></mxCell>'''

    # 4 Viewers
    xml += f'''<mxCell id="vw" value="Viewers illimites<br>HTTP-FLV<br>ABR Auto" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EF4444;strokeColor=#DC2626;fontColor=#FFFFFF;shadow=1;arcSize=8;fontSize=12;" vertex="1" parent="1"><mxGeometry x="1360" y="180" width="160" height="120" as="geometry"/></mxCell>'''

    # Arrows
    arr = "edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#94A3B8;strokeWidth=2;html=1;"
    xml += f'''
    <mxCell id="a1" style="{arr}" edge="1" parent="1" source="ios" target="ha"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a2" style="{arr}" edge="1" parent="1" source="ha" target="srs"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a3" style="{arr}" edge="1" parent="1" source="srs" target="qsv"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a4" style="{arr}" edge="1" parent="1" source="qsv" target="edg"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a5" style="{arr}" edge="1" parent="1" source="edg" target="cdn"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="a6" style="{arr}" edge="1" parent="1" source="cdn" target="vw"><mxGeometry relative="1" as="geometry"/></mxCell>
    '''

    xml += f'''<mxCell id="mon" value="MONITORING: {servers['mon']}" style="rounded=1;fillColor=#1E293B;fontColor=#F8FAFC;strokeColor=#0F172A;fontSize=12;fontStyle=1;arcSize=6;" vertex="1" parent="1"><mxGeometry x="40" y="560" width="1500" height="40" as="geometry"/></mxCell>'''

    xml += FOOTER
    with open(filename, 'w') as f: f.write(xml)

def create_roadmap():
    filename = "diagrams/scaling-roadmap.drawio"
    xml = HEADER.format(diag_id="roadmap", diag_name="Scaling Roadmap")
    xml += f'''
        <mxCell id="title" value="ShopFeed Live Streaming - Scaling Roadmap" style="text;html=1;align=center;verticalAlign=middle;fontSize=28;fontStyle=1;fontColor=#1E293B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="30" width="1000" height="40" as="geometry"/></mxCell>
        <mxCell id="subtitle" value="De 45 a 2000+ lives simultanes" style="text;html=1;align=center;fontSize=14;fontColor=#64748B;strokeColor=none;fillColor=none;" vertex="1" parent="1"><mxGeometry x="300" y="70" width="1000" height="20" as="geometry"/></mxCell>
    '''

    phases = [
        ("PHASE 1A", "45 Lives", "Demarrage Rapide", 60, "#F0FDF4", "#16A34A", [
            "LB: 2x CX33 HA", "Origin: 1x Rise-1", "QSV: 3+1 EX44", "Edge: 1x Rise-2", 
            "Redis: CPX11", "Monitoring: CX33"
        ]),
        ("PHASE 1 MVP", "300 Lives", "Validation Produit", 440, "#EFF6FF", "#2563EB", [
            "LB: 2x CX33 HA", "Origin: 1x Rise-1", "QSV: 20+1 EX44", "Edge: 3x Rise-2",
            "Redis: CPX11", "Monitoring: AX41"
        ]),
        ("PHASE 2", "1000 Lives", "Scale Massif", 820, "#FFF7ED", "#EA580C", [
            "LB: 2x CX33 HA", "Origin: 2x Rise-1 HA", "QSV: 50+ EX44", "Edge: 3x Rise-2 2Gbps",
            "Redis: 2x CPX32 HA", "Monitoring: AX41"
        ]),
        ("PHASE 3", "2000+ Lives", "Production Critique", 1200, "#FEF2F2", "#DC2626", [
            "LB: 2x CX33 HA", "Origin: 2x Rise-1 HA", "QSV: 88-117 EX44", "Edge: 4+ Rise-2 2Gbps",
            "Redis: 2x CPX32 HA", "Monitoring: 2x AX41 HA", "P2P CDN Streamroot"
        ])
    ]

    for title, lives, desc, x, fill, stroke, items in phases:
        xml += f'''<mxCell id="bg_{x}" value="" style="rounded=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;arcSize=4;" vertex="1" parent="1"><mxGeometry x="{x}" y="130" width="340" height="420" as="geometry"/></mxCell>'''
        xml += f'''<mxCell id="t1_{x}" value="{title}" style="text;html=1;align=center;fontStyle=1;fontSize=18;fontColor={stroke};" vertex="1" parent="1"><mxGeometry x="{x}" y="150" width="340" height="30" as="geometry"/></mxCell>'''
        xml += f'''<mxCell id="t2_{x}" value="{lives}" style="text;html=1;align=center;fontStyle=1;fontSize=24;fontColor=#1E293B;" vertex="1" parent="1"><mxGeometry x="{x}" y="180" width="340" height="40" as="geometry"/></mxCell>'''
        xml += f'''<mxCell id="t3_{x}" value="{desc}" style="text;html=1;align=center;fontSize=14;fontColor=#64748B;" vertex="1" parent="1"><mxGeometry x="{x}" y="220" width="340" height="30" as="geometry"/></mxCell>'''

        y_item = 270
        for item in items:
            xml += f'''<mxCell id="item_{x}_{y_item}" value="{item}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontColor=#1E293B;shadow=1;arcSize=8;fontSize=13;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="{x+40}" y="{y_item}" width="260" height="35" as="geometry"/></mxCell>'''
            y_item += 45

    xml += FOOTER
    with open(filename, 'w') as f: f.write(xml)

# Execute
create_global_arch()

create_phase(
    "1a", "Phase 1A - 45 Lives", "Demarrage rapide avec haute disponibilite",
    {
        "lb": "HAProxy LB<br>2x CX33<br>Keepalived VIP Active/Passive",
        "origin": "SRS Origin<br>1x OVH Rise-1",
        "redis": "Redis Single Node<br>1x CPX11",
        "qsv": "INTEL QSV CLUSTER<br>3 Active + 1 Standby EX44<br>| 45 streams totaux |",
        "edge": "SRS Edge<br>1x OVH Rise-2",
        "cdn": "Bunny CDN + Gcore",
        "mon": "1x CX33 (Prometheus, Grafana, Loki)"
    }
)

create_phase(
    "1-mvp", "Phase 1 MVP - 300 Lives", "Validation produit et stabilite",
    {
        "lb": "HAProxy LB<br>2x CX33<br>Keepalived VIP",
        "origin": "SRS Origin<br>1x OVH Rise-1",
        "redis": "Redis Single Node<br>1x CPX11",
        "qsv": "INTEL QSV CLUSTER<br>20 Active + 1 Standby EX44<br>| 300 streams totaux |",
        "edge": "SRS Edge<br>3x OVH Rise-2 (1Gbps)",
        "cdn": "Bunny CDN + Gcore",
        "mon": "1x AX41 NVMe (Monitoring + Alerting)"
    }
)

create_phase(
    "2-scale", "Phase 2 - 1000 Lives", "Scale up de l'infrastructure",
    {
        "lb": "HAProxy LB<br>2x CX33<br>Keepalived VIP",
        "origin": "SRS Origin HA<br>2x OVH Rise-1 (Primary/Backup)",
        "redis": "Redis HA Subsystem<br>2x CPX32",
        "qsv": "INTEL QSV CLUSTER<br>50+ EX44 Servers<br>| Auto-scaling scripts actifs |<br>| Buffer de 5 serveurs |",
        "edge": "SRS Edge HA<br>3x OVH Rise-2 (2Gbps vRack)",
        "cdn": "Cloudflare GeoDNS + Multi-CDN",
        "mon": "1x AX41 NVMe (Monitoring + Alerting + Scaling Advisor)"
    }
)

create_phase(
    "3-production", "Phase 3 - 2000+ Lives", "Haute performance et redondance totale",
    {
        "lb": "HAProxy LB<br>2x CX33<br>Keepalived VIP",
        "origin": "SRS Origin HA<br>2x OVH Rise-1",
        "redis": "Redis HA Subsystem<br>2x CPX32",
        "qsv": "INTEL QSV MEGA CLUSTER<br>88 to 117 EX44 Servers<br>| Organises en Racks Virtuels |<br>| 10 serveurs Hot Standby |",
        "edge": "SRS Edge Cluster<br>4+ OVH Rise-2 (2Gbps vRack)",
        "cdn": "Multi-CDN + P2P CDN (Streamroot)",
        "mon": "2x AX41 NVMe HA (Monitoring + Alerting + Scaling Advisor)"
    }
)

create_roadmap()

print("All diagrams regenerated gracefully!")
