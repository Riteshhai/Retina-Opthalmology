"""Typed, local aggregate research discovery and development-only simulation."""
from dataclasses import asdict
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
import numpy as np

from api.schemas import (ExperimentList, ExperimentMetrics, ExperimentSummary, PolicyOptions,
                         SimulationRequest, SimulationResponse, TriageConfiguration)
from qsart.data.idrid_provenance import idrid_source_confirmation
from qsart.triage_v2 import (Action, BudgetOverloaded, METHODS, PolicyConfig,
                             allocate_rows, screening_metrics)


DIAGNOSTIC_METRICS = {
    'mshf_quality_v1': ('inventory', 'overlap', 'model', 'populations', 'archive',
                        'source_commit', 'protocol_sha256'),
    'referability_gate_v1': ('images', 'patients', 'calibration', 'policy', 'budget_pairs',
                             'fits', 'default_ordinal_sets_replayed', 'input_sha256',
                             'source_commit', 'protocol_sha256'),
}


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
        ('spec_v1/efficientnet_selection.json', 'selection_dr_efficientnet', 'validation', 'development_selection'),
        ('robustness_v1/evaluation.json', 'controlled_degradation_v1', 'validation', 'development_model_input_stress'),
        ('ordinal_study_v1/evaluation.json', 'ordinal_randomization_v1', 'validation', 'development_ordinal_diagnostics'),
        ('repeated_seeds_v1/summary.json', 'repeated_seeds_v1', 'validation', 'development_selection'),
        ('policy_comparison_v1/dashboard_summary.json', 'policy_comparison_v1', 'validation',
         'development_out_of_fold_policy'),
        ('aptos_v1/evaluation.json', 'aptos_development_v1', 'validation',
         'development_selection_and_independent_calibration'),
        ('aptos_finetune_v1/evaluation.json', 'aptos_finetune_v1', 'validation',
         'development_finetuning_and_training_seed_diagnostics'),
        ('aptos_optimization_v2/evaluation.json', 'aptos_optimization_v2', 'validation',
         'source_development_primary_seed_schedule_extension'),
        ('aptos_384_v1/evaluation.json', 'aptos_384_v1', 'validation',
         'source_development_384px_imagenet_finetuning'),
        ('aptos_224_matched_v1/evaluation.json', 'aptos_224_matched_v1', 'validation',
         'source_development_matched_resolution_224_vs_384'),
        ('aptos_resolution_seed2027_v1/evaluation.json', 'aptos_resolution_seed2027_v1', 'validation',
         'source_development_matched_resolution_repeat'),
        ('quality_adaptation_v1/evaluation_v1_1.json', 'quality_adaptation_v1', 'validation',
         'development_matched_quality_backbone_adaptation'),
        ('quality_calibration_stress_v1/evaluation.json', 'quality_calibration_stress_v1', 'calibration_and_validation',
         'development_quality_temperature_and_input_stress'),
        ('eyeq_data_v1/data_audit.json', 'eyeq_data_v1', 'external_candidate_unscored',
         'external_quality_annotation_and_access_readiness_only'),
        ('adaptive_sets_v1/evaluation.json', 'adaptive_sets_v1', 'validation',
         'reused_development_fixed_aps_raps_score_comparison'),
        ('idrid_data_v1/data_audit.json', 'idrid_data_v1', 'external_holdout',
         'external_data_integrity_and_overlap_readiness_only'),
        ('idrid_review_v1/data_review.json', 'idrid_review_v1', 'external_holdout',
         'external_overlap_review_and_population_freeze'),
        ('idrid_external_v1/evaluation.json', 'idrid_external_v1', 'external_holdout',
         'locked_external_domain_diagnostics_consumed'),
        ('referability_gate_v1/evaluation.json', 'referability_gate_v1', 'validation',
         'development_referability_gate_diagnostic'),
        ('mshf_quality_v1/evaluation.json', 'mshf_quality_v1', 'external',
         'external_overall_quality_diagnostic')]
    for relative, name, partition, scope in sources:
        if any(record.experiment_id == name for record in records):
            continue
        path = root/relative
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding='utf-8'))
        metric_keys = ('grading', 'quality', 'classical_quality', 'conformal',
                   'budgets', 'images', 'patients', 'selection_metrics', 'conditions', 'paired_changes',
                   'condition_parameters', 'clean_equivalence', 'comparisons', 'randomization_summaries',
                   'deterministic_score_diagnostics', 'baseline_reproduction', 'summary', 'runs',
                   'population', 'head_rng_version', 'scenarios', 'paired_comparisons',
                   'evaluation_ledger', 'source_report_sha256', 'paired_bootstrap',
                   'decision_review_prices', 'mode_interpretation', 'status_interpretation',
                   'candidates', 'selected_candidate', 'selection_rule', 'temperature',
                   'calibration_fit', 'data_preparation', 'grading_before_temperature',
                   'calibration', 'calibration_unit_definition', 'uniform_generation', 'baseline',
                   'source_provenance_status', 'official_partitions', 'images_decoded', 'reference_datasets',
                   'cross_source_candidate_pairs', 'cross_source_candidate_images', 'cross_source_exact_pairs',
                   'internal_candidate_pairs', 'internal_label_conflict_components',
                   'independence_claim_blocked_by_detected_overlap', 'observations_excluded',
                   'model_inference_performed', 'external_model_evaluation_completed',
                   'source_confirmation', 'audit_source_provenance_status', 'models',
                   'paired_baseline_changes', 'external_outcomes_consumed',
                   'classifier_fitting_performed', 'temperature_fitting_performed',
                   'conformal_threshold_fitting_performed', 'prior_primary', 'primary_seed',
                   'fixed_epochs_per_run', 'majority_references', 'warm_start_reproduction',
                   'production_model_changed', 'defect_mapping', 'primary_model_id',
                   'primary_calibration_metric', 'primary_stress_conditions', 'primary_stress_metric',
                   'response_contrasts', 'bootstrap', 'interval_counts',
                   'corrupted_quality_ground_truth_available', 'annotation_profile',
                   'count_reference_differences', 'access_review', 'readiness', 'storage_snapshot',
                   'author_source_revision', 'images_downloaded', 'configuration', 'fixed_temperature',
                   'model_training_seed', 'raps_parameter_tuning_performed',
                   'matched_reference', 'fixed_epoch_comparison')
        metrics = {key: raw[key] for key in DIAGNOSTIC_METRICS.get(name, metric_keys) if key in raw}
        if name.startswith('selection_') and 'selection_metrics' in raw:
            metrics = raw['selection_metrics'].get('dr_grade', raw['selection_metrics'])
        if name == 'idrid_data_v1':
            metrics.update(idrid_source_confirmation(path))
        model_info = raw.get('model' if name == 'mshf_quality_v1' else 'model_info', {})
        records.append(ExperimentSummary(experiment_id=name, dataset=raw.get('dataset', 'DeepDRiD'), partition=partition,
            scope=scope, model_version=model_info.get('model_version', raw.get('model_version')),
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
