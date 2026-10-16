"""Verify the allowlisted publication snapshot without opening private datasets."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.py', '.ipynb', '.csv', '.json', '.md', '.txt'}
FORBIDDEN_COLUMNS = {'source_row', 'profile_group', 'patient_id', 'encounter_id',
                     'subject_id', 'probability', 'oof_pred_proba', 'y'}
SECRET = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|AKIA[0-9A-Z]{16}')

def fail(message):
    errors.append(message)

errors = []
manifest_path = ROOT / 'RELEASE_MANIFEST.csv'
if not manifest_path.exists():
    sys.exit('FAIL: release manifest is absent')
with manifest_path.open(encoding='utf-8', newline='') as f:
    manifest = list(csv.DictReader(f))
expected = {row['path']: row for row in manifest}
files = [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts]
for p in files:
    rel = p.relative_to(ROOT).as_posix()
    if rel == 'RELEASE_MANIFEST.csv':
        continue
    if rel not in expected:
        fail('Unmanifested file: '+rel)
        continue
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected[rel]['sha256']:
        fail('Checksum mismatch: '+rel)
    if len(data) != int(expected[rel]['bytes']):
        fail('Size mismatch: '+rel)
    if p.suffix not in ALLOWED and p.name != '.gitignore':
        fail('Unapproved file type: '+rel)
    if len(data) > 10_000_000:
        fail('Unexpected large file: '+rel)
    text = data.decode('utf-8-sig')
    if SECRET.search(text):
        fail('Potential credential: '+rel)
    if re.search(r'[A-Za-z]:[\\/]Users[\\/]', text):
        fail('Personal absolute path: '+rel)
    if p.suffix == '.ipynb':
        nb = json.loads(text)
        for c in nb['cells']:
            if c.get('outputs') or c.get('execution_count') is not None or c.get('attachments'):
                fail('Notebook contains outputs, execution counts or attachments: '+rel)
    if rel.startswith(('Results/', 'Registry/')) and p.suffix == '.csv':
        rows = list(csv.DictReader(text.splitlines()))
        fields = set(rows[0]) if rows else set()
        if fields & FORBIDDEN_COLUMNS:
            fail('Potential record-level columns: '+rel)
        for row in rows:
            counts = {}
            for key in ['n', 'events', 'positives', 'negatives', 'hidden_test_n', 'hidden_test_events']:
                try:
                    counts[key] = float(row[key])
                except (KeyError, ValueError, TypeError):
                    continue
                if 0 < counts[key] < 10:
                    fail('Small explicit group/outcome count: '+rel)
            for n, events in [('n','events'), ('hidden_test_n','hidden_test_events')]:
                if n in counts and events in counts and 0 < counts[n]-counts[events] < 10:
                    fail('Small complementary outcome count: '+rel)
for rel in expected:
    if not (ROOT/rel).is_file():
        fail('Missing manifest file: '+rel)
if errors:
    print('\n'.join(sorted(set(errors))))
    sys.exit(1)
print(f'PASS: {len(expected)} manifest entries; no flagged patterns or small explicit counts.')
print('This is a disclosure/integrity screen, not data-use approval or experimental reproduction.')
