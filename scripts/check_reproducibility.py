"""Compare this run's metrics with the reference values quoted in the v3 report."""
import json, os
REF = {'A_flat_L2': (0.388, -20.6), 'A_flat_L1': (0.416, -21.2), 'A_topo_L2': (0.357, -11.5), 'A_topo_L1': (0.345, -9.7),
       'A_topo_L1_z05': (0.338, -6.8), 'A_topo_L1_z2': (0.348, -8.3), 'A_topo_L1_z025': (0.334, -6.9),
       'A_flat_L1_z05': (0.407, -20.0), 'A_flat_L1_z2': (0.407, -20.2), 'A_flat_L1_z025': (0.401, -21.0),
       'N5_topo': (0.350, -10.0), 'N5_flat': (0.421, -24.7)}
bad = 0
print(f'{"run":16s} {"log-misfit ref/now":>20s} {"dx ref/now (m)":>18s}')
for k, (lm, dx) in REF.items():
    f = f'met_{k}.json'
    if not os.path.exists(f): print(f'{k:16s} MISSING'); bad += 1; continue
    m = json.load(open(f)); ok = abs(m['rms_log'] - lm) < 0.005 and abs(m['dx'] - dx) < 0.5
    bad += not ok
    print(f'{k:16s} {lm:8.3f} / {m["rms_log"]:.3f}   {dx:7.1f} / {m["dx"]:6.1f}   {"OK" if ok else "DIFFERS"}')
ip = {'A_topo_L1_z05': {'coincident': (-5, -3), 'offset': (83, 23)}, 'A_flat_L1_z05': {'coincident': (0, -1), 'offset': (92, 25)}}
for k, sc in ip.items():
    m = json.load(open(f'ipmet_{k}.json'))
    for s, (dx, dz) in sc.items():
        o = m[s]['recovered_offset']; ok = abs(o[0] - dx) < 1.5 and abs(o[1] - dz) < 1.5; bad += not ok
        print(f'IP {k:14s} {s:10s} ref ({dx:+d},{dz:+d})  now ({o[0]:+.0f},{o[1]:+.0f})  {"OK" if ok else "DIFFERS"}')
print('\nREPRODUCIBILITY:', 'ALL MATCH' if bad == 0 else f'{bad} item(s) differ')
