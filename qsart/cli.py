"""Usage: python -m qsart.cli --demo OR --calibration file.csv --test file.csv"""
import argparse
import json
from pathlib import Path
import pandas as pd
from .policy import decide, tune, metrics, validate
from .synthetic import make_mock


def evaluate(calibration, test, budgets):
    validate(calibration); validate(test)
    overlap = set(calibration.patient_id.astype(str)) & set(test.patient_id.astype(str))
    if overlap: raise ValueError(f'Patient leakage: {len(overlap)} overlapping IDs')
    # Independently tune every policy under exactly the same feasible budgets.
    # This is budget-matched, not necessarily equal *actual* review/recapture rates;
    # publication claims still require operating-point / Pareto-matched comparisons.
    fitted, test_results, routes, infeasible = {}, {}, {}, {}
    for method in ('qsart', 'quality_first', 'confidence_only', 'risk_only'):
        try:
            t, fit = tune(calibration, method=method, **budgets)
            fitted[method] = {'thresholds': t.as_dict(), 'calibration': fit}
            actions = decide(test, t, method)
            test_results[method] = metrics(test, actions)
            routes[method] = actions
        except ValueError as e:
            infeasible[method] = str(e)
    if 'qsart' not in fitted:
        raise ValueError('QSART infeasible under the requested budgets; choose a documented scenario, do not silently relax it.')
    report = {
        'important': 'RESEARCH BENCH ONLY / NOT A DIAGNOSTIC MODEL',
        'budgets': budgets,
        'fits': fitted,
        'test': test_results,
        'infeasible': infeasible,
        'comparison_warning': ('Baselines independently tuned using the same calibration-only constraints, '
                               'but their actual review/recapture rates are not necessarily matched. '
                               'Do not claim superiority without operating-point matching, '
                               'paired confidence intervals and independent clinical data.')}
    return report, pd.DataFrame({
        'image_id': test.image_id, 'action': routes['qsart'], 'grade': test.grade,
        'quality': test.quality, 'p_rdr': test.p_rdr, 'uncertainty': test.uncertainty})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo',action='store_true')
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--test',type=Path)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--max-review',type=float,default=.50)
    parser.add_argument('--max-recapture',type=float,default=.30)
    parser.add_argument('--min-routine',type=float,default=.18)
    args=parser.parse_args()
    if args.demo:
        calibration=make_mock(n=1000,seed=42,prefix='cal')
        test=make_mock(n=1000,seed=43,prefix='test')
    else:
        if not(args.calibration and args.test): parser.error('Supply --demo or both --calibration and --test')
        calibration=pd.read_csv(args.calibration)
        test=pd.read_csv(args.test)
    result, routes = evaluate(calibration,test,{
        'max_review':args.max_review,'max_recapture':args.max_recapture,'min_routine':args.min_routine})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    routes.to_csv(args.output/'test_routes.csv',index=False)
    if args.demo:
        calibration.to_csv(args.output/'mock_calibration.csv',index=False)
        test.to_csv(args.output/'mock_test.csv',index=False)
    print('Saved research-only outputs to:',args.output.resolve())
    print(json.dumps(result['test'],indent=2))

if __name__=='__main__': main()
