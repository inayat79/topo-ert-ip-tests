"""
run_all.py — runs the complete synthetic pipeline in the correct order.
Works on Windows, macOS, Linux, Jupyter (%run run_all.py) and Colab (!python run_all.py).
Resumable: a step is skipped if its output file already exists (delete the file to redo it).
Every step's output is written to ../logs/<step>.txt . Total time roughly 30-40 min on one CPU core.
"""
import subprocess, sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
LOGS = os.path.join(HERE, '..', 'logs'); os.makedirs(LOGS, exist_ok=True)
PY = sys.executable
# (log name, command args, output file that marks completion)
STEPS = [
 ('t01_validate',       ['t01_validate.py'],                                   None),
 ('t02_meshconv',       ['t02_meshconv.py'],                                   None),
 ('t03_topo_effect',    ['t03_topo_effect.py'],                                'corr.npy'),
 ('t04_topo_conv',      ['t04_topo_convergence.py'],                           None),
 ('t05_make_data',      ['t05_make_data.py'],                                  'ma_noise_offset.npy'),
]
for tag, topo, rob, zw in [('A_flat_L2',0,0,1.0),('A_flat_L1',0,1,1.0),('A_topo_L2',1,0,1.0),('A_topo_L1',1,1,1.0),
                           ('A_topo_L1_z05',1,1,0.5),('A_topo_L1_z2',1,1,2.0),('A_topo_L1_z025',1,1,0.25),
                           ('A_flat_L1_z05',0,1,0.5),('A_flat_L1_z2',0,1,2.0),('A_flat_L1_z025',0,1,0.25)]:
    STEPS.append((f'inv_{tag}', ['t06_invert.py', tag, 'rhoa_noise3.npy', str(topo), str(rob), str(zw), '0.03'], f'res_{tag}.npz'))
STEPS += [
 ('inv_N5_topo',  ['t06_invert.py', 'N5_topo', 'rhoa_noise5.npy', '1', '1', '0.5', '0.05'], 'res_N5_topo.npz'),
 ('inv_N5_flat',  ['t06_invert.py', 'N5_flat', 'rhoa_noise5.npy', '0', '1', '0.5', '0.05'], 'res_N5_flat.npz'),
 ('t07_plot_A',   ['t07_plot_A.py'],                  'F03_testA_sections.png'),
 ('ip_topo',      ['t08_ip.py', 'A_topo_L1_z05'],     'ipmet_A_topo_L1_z05.json'),
 ('ip_flat',      ['t08_ip.py', 'A_flat_L1_z05'],     'ipmet_A_flat_L1_z05.json'),
 ('t10_plots',    ['t10_plots.py'],                   'F05_ip_offset_test.png'),
 ('t13_geomB',    ['t13_geometryB.py'],               'met_B_topo.json'),
 ('t15_plot_B',   ['t15_plot_B.py'],                  'F08_geometryB.png'),
 ('t16_geometryC', ['t16_geometryC.py'],               'met_C_topo.json'),
]
only = sys.argv[1:]            # optional: run only named steps, e.g.  python run_all.py t01_validate
t0 = time.time(); failed = []
for name, args, marker in STEPS:
    if only and name not in only: continue
    if marker and os.path.exists(marker):
        print(f'[skip] {name} (found {marker})'); continue
    print(f'[run ] {name} ...', flush=True); t = time.time()
    with open(os.path.join(LOGS, name + '.txt'), 'w') as lf:
        rc = subprocess.call([PY] + args, stdout=lf, stderr=subprocess.STDOUT)
    print(f'       {"OK" if rc == 0 else "FAILED (see logs/" + name + ".txt)"}  {time.time()-t:.0f}s', flush=True)
    if rc: failed.append(name); break          # stop at first failure so later steps don't use bad inputs
print(f'\nFinished in {(time.time()-t0)/60:.1f} min.', 'FAILED: ' + ', '.join(failed) if failed else 'All steps OK.')
