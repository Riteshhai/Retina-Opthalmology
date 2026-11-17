# Implemented research policy

For image i, the DR model supplies a five-grade probability vector p_i, the
quality model supplies gradability q_i, and normalized entropy supplies u_i.
The provisional grade is argmax p_i. AUTO_SCREEN can therefore report a positive
DR grade; it does not mean disease-free. The retained starter's `routine` outcome
uses a different auto-clearance definition and is a separate legacy benchmark.

For a cohort of N images, introduce binary assignments x_(i,a) to AUTO_SCREEN,
RECAPTURE or SPECIALIST_REVIEW. Minimize sum_i,a c_(i,a) x_(i,a), with exactly one
action per image and upper bounds floor(N B_review) and floor(N B_recapture).
The transportation LP has an integral assignment polytope. Implementation checks
integrality and both realized integer capacities before returning results.

The exploratory auto cost is expected severity-weighted grade error under p_i,
plus (1-q_i) and 0.5u_i. The provisional grade used for that expectation is fixed
before allocation. Weights [1,1,3,6,10] emphasize higher reference grades; they
are assumptions for sensitivity analysis, not established clinical costs.
The review cost is 0.25. The recapture proxy is 0.25 + expected severity weight
+ 0.5q_i. It discourages delaying high-risk cases for recapture, without assuming
that another photograph improves quality or that a reviewer diagnoses correctly.

AUTO_SCREEN is forbidden below quality 0.5. If insufficient review/recapture
capacity makes that condition impossible, allocation raises BudgetInfeasible.
The API reports HTTP 422; budgets are never silently enlarged. A single-image
request uses fixed risk/quality/entropy thresholds and explicitly carries no
population workload guarantee. These fixed exploratory settings were declared
before the pilot holdout was evaluated, not tuned on holdout outcomes.

Seven policies are available: plain grading; confidence only; quality only;
severity only; single review budget; naive quality recapture; dual budget.
They share requested capacity limits, but can have different actual workloads.
Plain grading deliberately has no quality restriction. An infeasible comparator
is reported instead of discarded. Matched workload or Pareto comparisons are
required before claiming a research improvement. No improvement is guaranteed.

The separate-budget ablation retains the sum of integer capacities and removes
their independent caps. It deliberately treats review and recapture slots as
fungible; this is an experimental resource assumption, not a workflow recommendation.
Its separate-cap violations remain reported. The severity ablation changes
allocation costs but retains evaluation weights, keeping errors comparable.

Metrics distinguish errors among automated predictions, grade>=2 cases deferred
to recapture, and grade>=3 recapture delays. AUTO_SCREEN sensitivity/specificity
use the automated subset as denominator; neither is called whole-cohort clinical
sensitivity. Missing denominators are null. No completed care outcomes are inferred.
