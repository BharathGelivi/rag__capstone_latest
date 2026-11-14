"""Build a deterministic, dependency-free navigation index without importing ML code."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src', 'configs', 'scripts', 'experiments', 'tests', 'tests_eval', 'ui', 'frontend/src')
EXTENSIONS = {'.py', '.tsx', '.ts', '.css', '.yaml'}
ARTIFACT_IO = {
    'src/rag_trace.py': (['retrieval result', 'generation result'], ['artifacts/rag_traces/<date>/trace_<id>.json']),
    'src/claims.py': (['claim metadata'], []),
    'src/claim_decomposer.py': (['RAGTrace'], ['CandidateClaimSet', 'artifacts/debug/<date>/']),
    'src/claim_verifier.py': (['CandidateClaimSet', 'RAGTrace', 'artifacts/chunk_registry.json'], ['VerificationSummary', 'artifacts/verification/TRACE_<id>.json']),
    'src/pipeline_state_analyzer.py': (['RAGTrace', 'claim set', 'VerificationSummary'], ['artifacts/pipeline_state_matrix/TRACE_<id>.json']),
    'src/root_cause_reasoner.py': (['PipelineStateMatrix'], ['artifacts/root_cause_analysis/TRACE_<id>.json']),
    'src/corrective_action_engine.py': (['RootCauseAnalysis', 'PipelineStateMatrix'], ['artifacts/corrective_action_plan/TRACE_<id>.json']),
    'src/report_builder.py': (['RAGTrace', 'PipelineStateMatrix', 'RootCauseAnalysis', 'CorrectiveActionPlan', 'VerificationSummary'], ['DiagnosticEvaluationReport']),
    'src/report.py': (['report JSON'], ['artifacts/reports/<id>.json']),
    'src/report_presenter.py': (['DiagnosticEvaluationReport'], ['Markdown', 'HTML', 'PDF bytes']),
    'src/api.py': (['trace request', 'artifacts/reports/<id>.json'], ['artifacts/rag_traces/<id>.json', 'artifacts/reports/<id>.json']),
}


def build_index(root=ROOT):
    paths = sorted({p for directory in DIRECTORIES for p in (root / directory).rglob('*')
                    if p.suffix in EXTENSIONS and '__pycache__' not in p.parts}
                   | set(root.glob('*.py')))
    entries = []
    for path in paths:
        text = path.read_text(encoding='utf-8-sig')
        relative = path.relative_to(root).as_posix()
        entry = dict(path=relative, sha256=hashlib.sha256(text.encode()).hexdigest(),
                     purpose='', classes=[], functions=[], public_apis=[], imports=[],
                     configuration_dependencies=[], artifact_references=[], endpoints=[],
                     depends_on=[], depended_on_by=[], tests=[])
        entry['artifact_inputs'], entry['artifact_outputs'] = ARTIFACT_IO.get(relative, ([], []))
        if path.suffix == '.py':
            tree = ast.parse(text, filename=relative)
            entry['purpose'] = ast.get_docstring(tree) or ''
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    entry['imports'].extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    package = list(path.parent.relative_to(root).parts)
                    prefix = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
                    module = '.'.join(part for part in (prefix, node.module) if part)
                    entry['imports'].append(module)
                    entry['imports'].extend(module + '.' + alias.name for alias in node.names if alias.name != '*')
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol = dict(name=node.name, line=node.lineno, purpose=ast.get_docstring(node) or '')
                    entry['classes' if isinstance(node, ast.ClassDef) else 'functions'].append(symbol)
                    if not node.name.startswith('_'):
                        entry['public_apis'].append(node.name)
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr in {'get', 'post', 'put', 'delete', 'patch'} and decorator.args:
                                entry['endpoints'].append(dict(method=decorator.func.attr.upper(),
                                                              route=ast.unparse(decorator.args[0]), handler=node.name))
            entry['configuration_dependencies'] = sorted(i for i in entry['imports'] if i.startswith('configs'))
            entry['artifact_references'] = sorted({s.value for s in ast.walk(tree)
                if isinstance(s, ast.Constant) and isinstance(s.value, str)
                and re.match(r'^(artifacts|db|data)/', s.value)})
        else:
            entry['imports'] = re.findall(r'from\s+[\'"]([^\'"]+)', text)
            entry['public_apis'] = re.findall(r'export\s+(?:default\s+)?(?:function|class|const|interface|type)\s+(\w+)', text)
        if not entry['purpose']:
            entry['purpose'] = next((s['purpose'] for s in entry['classes'] + entry['functions'] if s['purpose']),
                                    'Configuration' if relative.startswith('configs/') else 'See symbols and dependencies; no source docstring.')
        entry['imports'] = sorted(set(entry['imports']))
        entries.append(entry)
    modules = {e['path'][:-3].replace('/', '.'): e for e in entries if e['path'].endswith('.py')}
    modules.update({name.removesuffix('.__init__'): entry for name, entry in list(modules.items())
                    if name.endswith('.__init__')})
    for entry in entries:
        for imported in entry['imports']:
            if imported in modules:
                target = modules[imported]
                if target['path'] not in entry['depends_on']:
                    entry['depends_on'].append(target['path'])
                    target['depended_on_by'].append(entry['path'])
                if entry['path'].startswith(('tests/', 'tests_eval/')):
                    if entry['path'] not in target['tests']:
                        target['tests'].append(entry['path'])
    concepts = {}
    for name in ('RAGTrace', 'ClaimSet', 'CandidateClaimSet', 'VerificationResult', 'PipelineStateMatrix',
                 'RootCauseAnalysis', 'CorrectiveActionPlan', 'DiagnosticEvaluationReport'):
        concepts[name] = [dict(path=e['path'], **s) for e in entries for s in e['classes'] if s['name'] == name]
    return dict(schema_version=1, limitations=[
        'Static references are navigation evidence, not proof of runtime test coverage.',
        'Artifact references include inputs and outputs; inspect save/load APIs for direction.',
        'Explicit artifact contracts are curated for the core diagnostic flow; empty lists mean unclassified, not no I/O. Save APIs may need explicit caller invocation.',
        'Dynamic imports, computed paths and re-exports may need manual tracing.'],
        concepts=concepts, files=entries)


if __name__ == '__main__':
    output = ROOT / 'docs' / 'semantic_index.json'
    output.parent.mkdir(exist_ok=True)
    index = build_index()
    output.write_text(json.dumps(index, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Indexed {len(index['files'])} files -> {output}")
