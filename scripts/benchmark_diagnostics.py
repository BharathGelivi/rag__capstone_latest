"""Offline diagnostic timing smoke test; optionally measure cached local NLI.

Run: python -m scripts.benchmark_diagnostics [--local-model]
No remote LLM calls. Synthetic evidence is explicitly excluded from model timings.
"""
import argparse
import json
import logging
import random
import statistics
import time


def measure(operation, repetitions=20):
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-model', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    from src.api import app
    from fastapi.testclient import TestClient
    from experiments.faults import build_healthy_case, diagnose
    from src.corrective_action_engine import CorrectiveActionEngine
    from src.report_builder import ReportBuilder
    from src.report_presenter import DiagnosticReportPresenter
    logging.disable(logging.INFO)
    timings = {'api_import_startup_ms': (time.perf_counter() - start) * 1000}
    case = build_healthy_case('PERF_SMOKE', random.Random(42))
    psm, rca = diagnose(case)
    cap = CorrectiveActionEngine().generate(rca, psm=psm)

    def build_report():
        return ReportBuilder().build(trace=case.trace, psm=psm, rca=rca, cap=cap,
                                     verification=case.verification)

    presenter = DiagnosticReportPresenter(build_report())
    with TestClient(app) as client:
        timings['health_request_median_ms'] = measure(lambda: client.get('/health').raise_for_status())
    timings['synthetic_state_and_cause_median_ms'] = measure(lambda: diagnose(case))
    timings['report_generation_median_ms'] = measure(build_report)
    timings['report_html_median_ms'] = measure(presenter.render_html)
    timings['model_initialization_ms'] = None
    timings['claim_verification_ms'] = None
    timings['live_end_to_end_ms'] = None
    if args.local_model:
        from src.claim_verifier import ClaimVerifier
        start = time.perf_counter()
        verifier = ClaimVerifier(enable_llm_judge=False)
        timings['model_initialization_ms'] = (time.perf_counter() - start) * 1000
        timings['claim_verification_ms'] = measure(
            lambda: verifier.run_nli('The punishment is life imprisonment.',
                                    'Life imprisonment is a punishment.'), repetitions=3)
    print(json.dumps({'timings': timings, 'limitations':
        'Startup is API import, not socket readiness. Health uses TestClient. '
        'Diagnostic timing uses synthetic preverified evidence, excludes retrieval and LLMs. '
        'Local NLI timing is one pair, not full claim/chunk verification. '
        'null means not measured; no live end-to-end claim is made.'}, indent=2))


if __name__ == '__main__':
    main()
