"""The configuration files agree with the code constants; the repository holds no data or outputs."""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_corpora_config_matches_the_code():
    from nextact import corpora as C
    from nextact.settings import lexical as F
    cfg = json.loads((REPO / 'configs' / 'corpora.json').read_text())
    for c, src in C.SOURCES.items():
        assert cfg[c]['source'] == src
        assert cfg[c]['sampled_dialogues'] == F.SAMPLE_N[c] and cfg[c]['data_fingerprint'] == F.FINGERPRINTS[c]['data']
        assert len(src['sha256']) == 64


def test_models_config_matches_the_code_with_full_revisions():
    from nextact import dense, encoder, icl
    cfg = json.loads((REPO / 'configs' / 'models.json').read_text())
    for k, v in icl.MODELS.items():
        assert cfg['language_models'][k]['repo'] == v['repo'] and cfg['language_models'][k]['revision'] == v['revision']
    assert cfg['encoder']['revision'] == encoder.ENC_SPEC['revision']
    assert cfg['dense_retriever']['revision'] == dense.DENSE['revision']
    revs = [m['revision'] for m in cfg['language_models'].values()] + [cfg['encoder']['revision'], cfg['dense_retriever']['revision']]
    assert all(len(r) == 40 and int(r, 16) >= 0 for r in revs)


def test_scientific_constants():
    from nextact import icl, protocol as P
    assert icl.K_DEMOS == 4 and icl.N_QUERIES == {'continuation': 300, 'transition': 450}
    assert (icl.N_BOOT, icl.BOOT_SEED) == (2000, 777) and P.N_BOOT == 2000
    assert P.PRESENCE_DESIGN == dict(n_repeats=5, n_folds=5, n_groups=4, base_seed=20261027)


def test_pipeline_dry_run_lists_every_module():
    r = subprocess.run([sys.executable, str(REPO / 'scripts' / 'reproduce.py'), '--dry-run'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for mod in ('lexical', 'encoder', 'knn', 'icl', 'icl_fresh', 'unlabelled', 'consistency', 'window_followup', 'dense',
                'exposure'):
        assert f'nextact.{mod} ' in r.stdout or f'nextact.{mod}\n' in r.stdout, mod


def test_repository_contains_no_data_outputs_or_binaries():
    tracked = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True).stdout.split()
    if not tracked:  # not a git checkout (e.g. an archive)
        tracked = [str(p.relative_to(REPO)) for p in REPO.rglob('*') if p.is_file() and not {'.git', '__pycache__', '.pytest_cache', 'work'} & set(p.parts)]
    allowed = ('.py', '.md', '.json', '.txt', '.toml', '.gitignore', '.cfg')
    bad = [f for f in tracked if not f.endswith(allowed) and f != 'LICENSE' and 'pytest_cache' not in f]
    assert bad == [], bad


def test_licence_is_mit_and_names_no_person():
    text = (REPO / 'LICENSE').read_text()
    assert text.startswith('MIT License') and 'Permission is hereby granted, free of charge' in text
    assert 'Copyright (c) 2026 next-act-exposure-repro contributors' in text


def test_every_import_resolves_to_the_package_stdlib_or_requirements():
    """Including imports inside functions: no module of the original research repository may be imported."""
    import ast
    std = set(sys.stdlib_module_names)
    req = {'numpy', 'pandas', 'scipy', 'sklearn', 'joblib', 'torch', 'transformers', 'huggingface_hub', 'pytest',
           'accelerate', 'safetensors', 'tokenizers'}
    bad = []
    for f in list((REPO / 'src').rglob('*.py')) + list((REPO / 'tests').glob('*.py')) + list((REPO / 'scripts').glob('*.py')):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            bad += [(f.name, n) for n in names if n not in std | req | {'nextact'}]
    assert bad == [], bad
