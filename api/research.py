"""Typed, local aggregate research discovery and development-only simulation."""
from dataclasses import asdict
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
import numpy as np

from api.schemas import (ExperimentList, ExperimentMetrics, ExperimentSummary, PolicyOptions,
                         SimulationRequest, SimulationResponse, TriageConfiguration)
from qsart.triage_v2 import (Action, BudgetOverloaded, METHODS, PolicyConfig,
                             allocate_rows, screening_metrics)


def experiment_registry(root):
    root = Path(root)
    records = []
    comparison_path = root/'spec_v1/model_comparison.json'
    if comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
        records.extend(ExperimentSummary(**row) for row in comparison['experiments'])
    lock_path = root/'development_v1/selection_lock.json'
    if lock_path.is_file() and not records:
        lock = json.loads(lock_path.read_text(encoding='utf-8'))
        for candidate in lock['candidates']:
            records.append(ExperimentSummary(experiment_id='selection_'+candidate['name'], dataset='DeepDRiD',
                partition='validation', scope='development_selection', model_version=candidate['name'],
                metrics={**candidate['metrics'], 'selected_epoch': candidate['selected_epoch']},
                notes=['Genuinely trained frozen-backbone candidate; selection data are not independent test evidence.']))
    sources = [
        ('confirmation_v1/evaluation.json', 'confirmation_v1', 'holdout', 'historical_internal_confirmation'),
        ('pilot/evaluation.json', 'pilot', 'holdout', 'historical_engineering_pilot'),
        ('spec_v1/development_evaluation.json', 'spec_v1_alignment', 'validation', 'development_exploration'),
        ('spec_v1/efficientnet_selection.json', 'selection_dr_efficientnet', 'validation', 'development_selection')]
    for relative, name, partition, scope in sources:
        if any(record.experiment_id == name for record in records):
            continue
        path = root/relative
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        metrics = {key: raw[key] for key in ('grading', 'quality', 'classical_quality', 'conformal',
                   'budgets', 'images', 'patients', 'selection_metrics') if key in raw}
        if name.startswith('selection_') and 'selection_metrics' in raw:
            metrics = raw['selection_metrics'].get('dr_grade', raw['selection_metrics'])
        records.append(ExperimentSummary(experiment_id=name, dataset='DeepDRiD', partition=partition,
            scope=scope, model_version=raw.get('model_info', {}).get('model_version', raw.get('model_version')),
            metrics=metrics, notes=raw.get('limitations', [
                'Historical AUTO_SCREEN triage is not the current AUTO_NONREFERABLE policy.'])))
    return records


def research_router(app):
    router = APIRouter()

    @router.get('/triage/config', response_model=TriageConfiguration)
    def config():
        return TriageConfiguration(actions=list(Action), defaults=PolicyOptions(**asdict(PolicyConfig())),
            methods=list(METHODS), simulation_datasets=['development_selection_v2'],
            compute_profiles={'single': 0., 'budget25': .25, 'all': 1.},
            notes=['All automatic outcomes are provisional non-referable clearances.',
                   'Mandatory deferrals are never relaxed to meet capacities.',
                   'Simulator uses development selection scores; no held-out final claim.',
                   'Compute profiles are precomputed requests; sliders do not rerun model inference.'])

    @router.post('/triage/simulate', response_model=SimulationResponse)
    def simulate(request: SimulationRequest):
        warnings = ['Exploratory simulation; labels/configurations here cannot establish held-out final performance.',
                    'Deferred recommendations are unresolved; costs and thresholds are research proxies.']
        try:
            options = PolicyConfig(**request.config.model_dump())
            truth = None
            if request.scores is not None:
                if request.dataset_id is not None or request.compute_profile != 'single':
                    raise ValueError('Supply scores or a cached development dataset; compute profiles require the cache')
                rows = [row.model_dump() for row in request.scores]
                dataset = 'supplied_scores'
                if all(row['true_grade'] is not None for row in rows):
                    truth = np.array([row['true_grade'] for row in rows], dtype=int)
                    warnings.append('Evaluation labels were supplied by the caller and are not source-verified.')
            else:
                dataset = request.dataset_id or 'development_selection_v2'
                source = os.environ.get('QSART_SIMULATION_DATA')
                if not source or not Path(source).is_file():
                    return SimulationResponse(status='unavailable', dataset_id=dataset,
                        warnings=['Real development score cache is unavailable; no simulation results were generated.'])
                cached = json.loads(Path(source).read_text(encoding='utf-8'))
                if (cached.get('dataset_id') != dataset or cached.get('partition') != 'validation'
                        or cached.get('scope') != 'development_exploration'):
                    raise ValueError('Only the reserved development selection cache is permitted')
                engine = app.state.engine
                if engine is None or cached['model_version'] != engine.model_version:
                    raise ValueError('Cached model version differs from the loaded model')
                if cached['dr_checkpoint_sha256'] != engine.calibration['dr']['checkpoint_sha256']:
                    raise ValueError('Cached model weights differ from the loaded model')
                if cached['quality_checkpoint_sha256'] != engine.calibration['quality']['checkpoint_sha256']:
                    raise ValueError('Cached quality weights differ from the loaded model')
                if app.state.conformal is None or cached['conformal_sha256'] != app.state.conformal.sha256:
                    raise ValueError('Cached conformal sidecar differs from the loaded model')
                selected = cached['profiles'][request.compute_profile]
                rows, truth = selected['rows'], np.asarray(cached['true_grades'], dtype=int)
                warnings.append('Compute scores were precomputed in fixed request chunks; budget changes do not rerun inference.')
            actions = allocate_rows(rows, options, request.method)
            if truth is not None:
                metrics = screening_metrics(truth, [row['grade_probabilities'] for row in rows], actions, options)
            else:
                metrics = {'coverage': actions.count(Action.AUTO_NONREFERABLE)/len(rows),
                           'review_rate': actions.count(Action.SPECIALIST_REVIEW)/len(rows),
                           'recapture_rate': actions.count(Action.RECAPTURE)/len(rows)}
            metrics['second_pass_fraction'] = sum(bool(row.get('second_pass_activated')) for row in rows)/len(rows)
            return SimulationResponse(status='evaluated', dataset_id=dataset, images=len(rows), actions=actions,
                metrics=metrics, compute_profile=request.compute_profile, warnings=warnings)
        except BudgetOverloaded as exc:
            return SimulationResponse(status='overloaded', dataset_id=locals().get('dataset', 'supplied_scores'),
                images=len(rows), diagnostics=exc.diagnostics, compute_profile=request.compute_profile,
                warnings=warnings+['No allocation was produced; automatic clearance was not forced.'])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            raise HTTPException(422, f'Invalid simulation input or incompatible development cache: {type(exc).__name__}') from None

    @router.get('/research/experiments', response_model=ExperimentList)
    def experiments():
        try:
            return ExperimentList(experiments=experiment_registry(os.environ.get('QSART_REPORTS_DIR', 'reports')))
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(503, 'Saved aggregate research reports could not be read') from None

    @router.get('/research/metrics', response_model=ExperimentMetrics)
    def metrics(experiment_id: str):
        record = next((row for row in experiments().experiments if row.experiment_id == experiment_id), None)
        if record is None:
            raise HTTPException(404, 'Experiment is not in the aggregate research registry')
        return ExperimentMetrics(experiment_id=record.experiment_id, scope=record.scope,
                                 metrics=record.metrics, notes=record.notes)

    return router
