"""Time the deployed CPU image-to-decision chain using real selection images."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import numpy as np

from qsart.models.checkpoint import sha256_file
from qsart.training.data import read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    parser.add_argument('--api', default='http://127.0.0.1:18000')
    parser.add_argument('--container', default='qsart-api-1')
    parser.add_argument('--output', type=Path, default=Path('reports/spec_v1/end_to_end_cpu.json'))
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Measured report exists; preserve it and choose a new output path')
    frame = read_manifest(args.manifest)
    sample = frame[frame.split == 'validation'].head(12)
    if len(sample) != 12:
        parser.error('This declared timing sample needs at least12 selection images')
    bodies = [Path(path).read_bytes() for path in sample.image_path]
    results = {}
    with httpx.Client(base_url=args.api, timeout=120) as client:
        info = client.get('/model/info').json()
        assert info['ready'] and info['device'] == 'cpu' and info['data_purpose'] == 'real_research'
        assert info['conformal']['status'] == 'available' and info['policy_version'] == '2.0'
        for second in (False, True):
            client.post('/predict', params={'second_pass': second},
                        files={'file': ('fundus.jpg', bodies[0], 'image/jpeg')}).raise_for_status()
        for name, batch_size, budget in [('single_first', 1, 0.), ('single_extra', 1, 1.), ('batch_budget25', 4, .25)]:
            elapsed, model_times, activated = [], [], []
            for start in range(0, len(bodies), batch_size):
                part = bodies[start:start+batch_size]
                before = time.perf_counter()
                if batch_size == 1:
                    response = client.post('/predict', params={'second_pass': bool(budget)},
                        files={'file': ('fundus.jpg', part[0], 'image/jpeg')})
                else:
                    response = client.post('/predict/batch', params={'review_budget': 1., 'recapture_budget': 1.,
                        'second_pass_budget': budget}, files=[('files', ('fundus.jpg', body, 'image/jpeg')) for body in part])
                response.raise_for_status()
                elapsed.append((time.perf_counter()-before)*1000)
                data = [response.json()] if batch_size == 1 else response.json()['results']
                assert all(row['conformal']['status'] == 'available' and row['policy_version'] == '2.0' for row in data)
                model_times.extend(row['processing_time_ms'] for row in data)
                activated.extend(row['second_pass_activated'] for row in data)
            results[name] = {'requests': len(elapsed), 'images': len(bodies), 'request_batch_size': batch_size,
                'mean_request_ms': statistics.mean(elapsed), 'p50_request_ms': statistics.median(elapsed),
                'p95_request_ms': float(np.percentile(elapsed, 95)),
                'amortized_wall_ms_per_image': sum(elapsed)/len(bodies),
                'mean_preprocessing_model_ms_per_image': statistics.mean(model_times),
                'second_pass_fraction': statistics.mean(activated), 'requested_second_pass_budget': budget}
    # Kernel high-water RSS for the actual API process, rather than a boundary sample.
    program = "import pathlib,json; s=pathlib.Path('/proc/1/status').read_text(); print(json.dumps({x.split(':')[0]:x.split(':')[1].strip() for x in s.splitlines() if x.startswith(('VmHWM:','VmRSS:'))})); p=pathlib.Path('/sys/fs/cgroup/memory.peak'); print(p.read_text().strip() if p.exists() else 'unavailable')"
    memory = {'scope': 'Actual API process/kernel high-water since container start, including startup and earlier verification'}
    observation = subprocess.run(['docker', 'exec', args.container, 'python', '-c', program], capture_output=True, text=True)
    if observation.returncode == 0:
        lines = observation.stdout.splitlines()
        memory['api_proc_status'] = json.loads(lines[0])
        memory['container_cgroup_memory_peak_bytes'] = int(lines[1]) if lines[1].isdigit() else None
    else:
        memory['status'] = 'unavailable'
    report = {'date_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Measured loopback HTTP CPU workflow',
        'model_version': info['model_version'], 'policy_version': '2.0', 'cpu_threads': info['cpu_threads'],
        'image_size': info['image_size'], 'model_forward_batch_size': 1, 'partition': 'validation',
        'sample_rule': 'First12 frozen selection-manifest rows; two warmup requests excluded',
        'manifest_sha256': sha256_file(args.manifest), 'source_sha256': sha256_file(__file__),
        'profiles': results, 'memory': memory,
        'limitations': ['Includes upload, JPEG decode, preprocessing, model, conformal, routing and JSON HTTP response.',
                        'Excludes client disk reads/browser rendering and remote-network effects.',
                        'Fixed sequential profile order on a live workstation; no architecture/strategy speed superiority claim.',
                        'Twelve images and three batch requests do not establish stable production-tail latency.',
                        'This is a timing study on development data, not clinical or external validation.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'profiles': results, 'memory': memory}, indent=2))


if __name__ == '__main__':
    main()
