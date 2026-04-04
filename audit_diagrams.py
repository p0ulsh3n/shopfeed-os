import re
import os

files = [
    'diagrams/shopfeed-live-infra.drawio',
    'diagrams/phase1a.drawio',
    'diagrams/phase1-mvp.drawio',
    'diagrams/phase2-scale.drawio',
    'diagrams/phase3-production.drawio',
    'diagrams/scaling-roadmap.drawio',
]

for f in files:
    with open(f, encoding='utf-8') as fh:
        content = fh.read()

    # Extract all vertex IDs (nodes)
    vertices = set(re.findall(r'<mxCell id="([^"]+)"[^>]*vertex="1"', content))
    # Also capture IDs that have vertex later
    vertices |= set(re.findall(r'<mxCell id="([^"]+)"[^/]*vertex="1"', content))

    # Extract all edges with their source and target
    edges = re.findall(r'<mxCell id="([^"]+)"[^>]*source="([^"]+)"[^>]*target="([^"]+)"', content)

    print()
    print("=== {} ===".format(os.path.basename(f)))
    print("  Nodes: {} | Edges: {}".format(len(vertices), len(edges)))

    broken = []
    for eid, src, dst in edges:
        src_ok = src in vertices
        dst_ok = dst in vertices
        if not src_ok or not dst_ok:
            broken.append("  [BROKEN] edge={} | src={}({}) -> dst={}({})".format(
                eid, src, "OK" if src_ok else "MISSING", dst, "OK" if dst_ok else "MISSING"
            ))

    if broken:
        print("  ISSUES FOUND:")
        for b in broken:
            print(b)
    else:
        print("  All edges verified: OK")

print()
print("Audit complete.")
