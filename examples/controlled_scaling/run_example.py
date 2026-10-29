"""Recompute the Scaling intervention point estimates from existing responses.

Run from the repository root with a fresh output directory outside the checkout:
    python -B examples/controlled_scaling/run_example.py --output-dir ../scaling-results
Requires Python 3.10+; uses only the standard library and makes no model calls.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent

def slope(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('Need paired values at two or more levels')
    if any(not math.isfinite(v) or v <= 0 for v in x+y):
        raise ValueError('Log-scale inputs must be finite and positive')
    lx, ly = [math.log10(v) for v in x], [math.log10(v) for v in y]
    mx, my = mean(lx), mean(ly)
    denominator = sum((v-mx)**2 for v in lx)
    if denominator == 0:
        raise ValueError('Population must vary within a setting')
    return sum((a-mx)*(b-my) for a,b in zip(lx,ly))/denominator

def analyze(rows, contract):
    selected = [r for r in rows if r['prompt_condition'] == contract['condition']]
    expected_ids = set(contract['setting_ids'])
    levels = set(contract['population_levels'])
    groups = {}
    for r in selected:
        key = r['setting_id']
        if key not in expected_ids:
            raise ValueError('Unknown setting: '+key)
        if r['request_status'] != 'valid':
            raise ValueError('Registered primary response is not valid: '+key)
        pop = float(r['population_people'])
        if pop not in levels:
            raise ValueError('Unknown Population level')
        group = groups.setdefault(key,{})
        if pop in group:
            raise ValueError('Duplicate setting/Population pair')
        group[pop] = r
    if set(groups) != expected_ids:
        raise ValueError('Missing registered setting')
    estimates = []
    for key in sorted(expected_ids):
        g = groups[key]
        if set(g) != levels:
            raise ValueError('Incomplete registered Population grid: '+key)
        x = sorted(levels)
        road = slope(x,[float(g[v]['road_length_km']) for v in x])
        gdp = slope(x,[float(g[v]['gdp_million_currency_units']) for v in x])
        estimates.append({'setting_id':key,'n_levels':len(x),'road_exponent':road,
                          'gdp_exponent':gdp,'gdp_minus_road':gdp-road})
    summary = {
        'scope':'Offline recomputation of existing primary-condition point estimates',
        'registered_settings':len(expected_ids),'analyzed_settings':len(estimates),
        'selected_rows':len(selected),'other_released_rows_not_analyzed_here':len(rows)-len(selected),
        'mean_road_exponent':mean(r['road_exponent'] for r in estimates),
        'mean_gdp_exponent':mean(r['gdp_exponent'] for r in estimates),
        'mean_difference':mean(r['gdp_minus_road'] for r in estimates),
        'settings_gdp_greater_than_road':sum(r['gdp_minus_road'] > 0 for r in estimates),
        'uncertainty':'This demonstration recomputes point estimates only; reported intervals remain in the original Source Data.',
    }
    return estimates,summary

def run(out):
    contract = json.loads((ROOT/'contract.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT/'input_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['inputs']:
        path = ROOT/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Input hash mismatch: '+entry['path'])
    with (ROOT/'data/urban_scaling_intervention_outputs.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    estimates,summary = analyze(rows,contract)
    with (ROOT/'data/figure4_source.csv').open(encoding='utf-8-sig',newline='') as f:
        reference = {r['contrast']:float(r['estimate']) for r in csv.DictReader(f) if r['benchmark']=='Urban Scaling'}
    for name,column in [('Road','mean_road_exponent'),('GDP','mean_gdp_exponent')]:
        if not math.isclose(summary[column],reference[name],rel_tol=0,abs_tol=1e-12):
            raise ValueError('Point estimate differs from original Figure 4 Source Data: '+name)
    if summary['settings_gdp_greater_than_road'] != contract['reported_gdp_greater_than_road']:
        raise ValueError('Directional count differs from the reported result')
    summary['matches_original_figure4_point_estimates'] = True
    summary['new_model_calls'] = 0
    out = Path(out).resolve()
    # Keep source inputs and the verified checkout unchanged.
    repository = ROOT.parents[1]
    if out == repository or repository in out.parents:
        raise ValueError('Choose an output directory outside the repository')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['summary.json','setting_estimates.csv']:
        if (out/name).exists():
            raise FileExistsError('Refuse to overwrite '+str(out/name))
    with (out/'setting_estimates.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(estimates[0]))
        writer.writeheader();writer.writerows(estimates)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output_dir),indent=2))
