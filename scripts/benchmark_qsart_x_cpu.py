"""Measure fixed-model CPU thread profiles on a single running Docker host."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from urllib.parse import urlsplit

import httpx


THREADS = (1, 2, 4)
REPEATS = 3
PHOTO_COUNT = 12
REFERENCE = 2
ABSOLUTE_TOLERANCE = 1e-6
MANIFEST_SHA256 = '45e9f2d34ed54a12056b57a808d03cfaf29ccc970d148e9eb107b79cf5ac3e0d'
MODEL_VERSION = 'development-v1-dr-mobilenet_v3_small-seed2026-epoch15'
ARTIFACTS = {
    'dr': ('QSART_DR_CHECKPOINT', 'f530056c27f83dbff8bba4ebb481cf7a9d82289947c4eda655a69ed22685a527'),
    'quality': ('QSART_QUALITY_CHECKPOINT', 'a600872ce8e2b256205350a5ef6c99b77d71f679d217c20c26810a3ed76eff43'),
    'calibration': ('QSART_CALIBRATION', '674821e445490c209fab99e56b0e9915ac406ca4ec8694e5efdbed63e456a737'),
    'model_card': ('QSART_MODEL_CARD', '43830695b8362aabd06df0d5dfa65b01086db9da2db3212001ac98b8b0e4710f'),
    'conformal': ('QSART_CONFORMAL', '15a98f6bfe93f81156c5592701995d08796dd889e3a5e43bcbde7ab5aa61c221'),
}
VOLATILE = {'request_id', 'processing_time_ms', 'inference_timestamp'}


@dataclass(frozen=True)
class Target:
    threads: int
    url: str
    container: str


@dataclass(frozen=True)
class Workload:
    name: str
    images_per_request: int
    second_pass_budget: float


WORKLOADS = (
    Workload('single_first', 1, 0.),
    Workload('single_extra', 1, 1.),
    Workload('batch_budget25', 4, .25),
)


def parse_target(value):
    try:
        count, url, container = value.split(',')
        threads = int(count)
        parsed = urlsplit(url)
        if (threads in THREADS and parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', '::1', 'localhost')
                and parsed.port and parsed.username is None and not parsed.query and not parsed.fragment
                and parsed.path in ('', '/') and container and container[0].isalnum()
                and all(ch.isascii() and (ch.isalnum() or ch in '_.-') for ch in container)):
            return Target(threads, url.rstrip('/'), container)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError('Expected THREADS,http://LOOPBACK:PORT,CONTAINER with THREADS in {1,2,4}')


def sample_manifest(path):
    manifest = path.read_bytes()
    if sha256(manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError('The timing manifest differs from the frozen Set-A development manifest')
    rows = csv.DictReader(StringIO(manifest.decode('utf-8-sig'), newline=''))
    sample = [row for row in rows if row['split'] == 'validation'][:PHOTO_COUNT]
    if (len(sample) != PHOTO_COUNT or any(row['annotation_status'] != 'verified' for row in sample)
            or len({row['sha256'] for row in sample}) != PHOTO_COUNT):
        raise ValueError('The frozen timing sample needs twelve distinct verified selection photos')
    data, hashes = [], []
    for row in sample:
        photo = (path.parent / row['image_path']).read_bytes()
        digest = sha256(photo).hexdigest()
        if digest != row['sha256'] or not photo.startswith(b'\xff\xd8'):
            raise ValueError('A timing photo differs from the frozen selection bytes')
        data.append(photo)
        hashes.append(digest)
    return data, sha256(''.join(hashes).encode('ascii')).hexdigest()


INSPECT_ARTIFACTS = r'''
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

def text_file(path):
    path = Path(path)
    return path.read_text().strip() if path.is_file() else None

def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

args = json.loads(sys.argv[1])
cpuinfo = text_file('/proc/cpuinfo') or ''
cpu = next((row.split(':', 1)[1].strip() for row in cpuinfo.splitlines() if row.startswith('model name')), None)
print(json.dumps({
    'artifacts': {name: digest(os.environ[env]) for name, env in args.items()},
    'runtime': {'machine': platform.machine(), 'cpu_model': cpu, 'cpu_affinity_count': len(os.sched_getaffinity(0)),
                'cpu_max': text_file('/sys/fs/cgroup/cpu.max'), 'python': platform.python_version(),
                'torch': version('torch'), 'torchvision': version('torchvision'), 'pillow': version('pillow')},
    'api_threads': int(os.environ['QSART_CPU_THREADS']),
}))
'''

INSPECT_MEMORY = r'''
import json
from pathlib import Path

if b'uvicorn' not in Path('/proc/1/cmdline').read_bytes():
    raise RuntimeError('Benchmark expects Uvicorn to be PID 1 in each API container')
stats = dict(line.split(':', 1) for line in Path('/proc/1/status').read_text().splitlines() if ':' in line)
fields = ('/sys/fs/cgroup/memory.peak', '/sys/fs/cgroup/memory/memory.max_usage_in_bytes')
peak = next((Path(path).read_text().strip() for path in fields if Path(path).is_file()), None)
print(json.dumps({'api_vm_hwm_kib': int(stats['VmHWM'].split()[0]),
                  'api_vm_rss_kib': int(stats['VmRSS'].split()[0]),
                  'container_cgroup_memory_peak_bytes': int(peak) if peak and peak.isdigit() else None}))
'''


def docker_json(container, program, *args):
    return json.loads(subprocess.check_output(['docker', 'exec', container, 'python', '-c', program, *args]))


def verify_targets(targets, clients):
    expected = {name: binding[1] for name, binding in ARTIFACTS.items()}
    image_ids, observed_runtimes = set(), []
    for target in targets:
        image_ids.add(subprocess.check_output(['docker', 'inspect', '--format', '{{.Image}}', target.container],
                                              text=True).strip())
        binding = docker_json(target.container, INSPECT_ARTIFACTS,
                              json.dumps({name: artifact[0] for name, artifact in ARTIFACTS.items()}))
        if binding['artifacts'] != expected or binding['api_threads'] != target.threads:
            raise ValueError('A target has different model artifacts or CPU threads')
        observed_runtimes.append(binding['runtime'])
        info = clients[target.threads].get('/model/info').raise_for_status().json()
        if (info['ready'] is not True or info['device'] != 'cpu' or info['data_purpose'] != 'real_research'
                or info['model_version'] != MODEL_VERSION or info['model_card_status'] != 'matched'
                or info['conformal']['artifact_sha256'] != expected['conformal'] or info['conformal']['status'] != 'available'
                or info['cpu_threads'] != target.threads or info['policy_version'] != '2.0'):
            raise ValueError('A target does not advertise the frozen ready research configuration')
    if len(image_ids) != 1 or any(observed != observed_runtimes[0] for observed in observed_runtimes):
        raise ValueError('The CPU comparison requires one Docker image and identical CPU visibility and limits')
    return {'docker_image_id': image_ids.pop(), 'artifact_sha256': expected, **observed_runtimes[0]}


@dataclass
class ResponseSample:
    payload: dict
    request_ms: float
    model_ms: list[float]
    activated: int


def measure_request(client, workload, images):
    started = time.perf_counter()
    if workload.images_per_request == 1:
        response = client.post('/predict', params={'second_pass': bool(workload.second_pass_budget)},
                               files={'file': ('fundus.jpg', images[0], 'image/jpeg')})
    else:
        response = client.post('/predict/batch', params={'review_budget': 1., 'recapture_budget': 1.,
                              'second_pass_budget': workload.second_pass_budget},
                               files=[('files', ('fundus.jpg', image, 'image/jpeg')) for image in images])
    elapsed = (time.perf_counter() - started) * 1000
    payload = response.raise_for_status().json()
    rows = [payload] if workload.images_per_request == 1 else payload['results']
    activated = sum(row['second_pass_activated'] for row in rows)
    times = [row['processing_time_ms'] for row in rows]
    if (len(rows) != len(images) or activated != math.floor(len(images) * workload.second_pass_budget)
            or any(row['conformal']['status'] != 'available' or row['policy_version'] != '2.0'
                   or row['model_version'] != MODEL_VERSION for row in rows)
            or any(not math.isfinite(value) or value < 0 for value in times)):
        raise ValueError('The target returned an unexpected workload, policy, or timing')
    return ResponseSample(payload, elapsed, times, activated)


def rotate(items, offset):
    offset %= len(items)
    return items[offset:] + items[:offset]


def collect_samples(clients, images):
    destination = {count: {profile.name: [] for profile in WORKLOADS} for count in THREADS}
    for threads in THREADS:
        for single in WORKLOADS[:2]:
            measure_request(clients[threads], single, images[:1])
    for repeat in range(REPEATS):
        for index, profile in enumerate(rotate(WORKLOADS, repeat)):
            for request, start in enumerate(range(0, len(images), profile.images_per_request)):
                batch = images[start:start + profile.images_per_request]
                for threads in rotate(THREADS, repeat + index + request):
                    measured = measure_request(clients[threads], profile, batch)
                    destination[threads][profile.name].append(measured)
    return destination


@dataclass
class Comparison:
    requests_compared: int = 0
    numeric_values_compared: int = 0
    max_absolute_numeric_difference: float = 0.
    numeric_tolerance_violations: int = 0
    semantic_violations: int = 0

    def compare(self, reference, candidate):
        if type(reference) is not type(candidate):
            self.semantic_violations += 1
        elif isinstance(reference, dict):
            fields = reference.keys() - VOLATILE
            if fields != candidate.keys() - VOLATILE:
                self.semantic_violations += 1
            for name in fields & candidate.keys():
                self.compare(reference[name], candidate[name])
        elif isinstance(reference, list):
            if len(reference) != len(candidate):
                self.semantic_violations += 1
            for first, second in zip(reference, candidate):
                self.compare(first, second)
        elif isinstance(reference, float):
            self.numeric_values_compared += 1
            if not math.isfinite(reference) or not math.isfinite(candidate):
                self.numeric_tolerance_violations += 1
            else:
                difference = abs(reference - candidate)
                self.max_absolute_numeric_difference = max(self.max_absolute_numeric_difference, difference)
                self.numeric_tolerance_violations += difference > ABSOLUTE_TOLERANCE
        else:
            self.semantic_violations += reference != candidate

    @property
    def passed(self):
        return not (self.numeric_tolerance_violations or self.semantic_violations)


def profile_summary(profile, samples):
    request = [sample.request_ms for sample in samples]
    model = [ms for sample in samples for ms in sample.model_ms]
    activated = sum(sample.activated for sample in samples)
    return {'requests': len(request), 'images': len(model), 'request_batch_size': profile.images_per_request,
            'mean_request_ms': statistics.mean(request), 'p50_request_ms': statistics.median(request),
            'p95_request_ms': statistics.quantiles(request, n=20, method='inclusive')[18],
            'amortized_wall_ms_per_image': sum(request) / len(model),
            'mean_preprocessing_model_ms_per_image': statistics.mean(model), 'second_passes': activated,
            'second_pass_fraction': activated / len(model), 'requested_second_pass_budget': profile.second_pass_budget}


def summarize_samples(samples, targets):
    rows, comparisons = {}, {}
    for target in targets:
        threads = target.threads
        rows[str(threads)] = {'cpu_threads': threads, 'profiles': {
            profile.name: profile_summary(profile, samples[threads][profile.name]) for profile in WORKLOADS},
            'memory': docker_json(target.container, INSPECT_MEMORY)}
        if threads == REFERENCE:
            continue
        totals = Comparison()
        for profile in WORKLOADS:
            reference, candidate = samples[REFERENCE][profile.name], samples[threads][profile.name]
            for first, second in zip(reference, candidate, strict=True):
                totals.requests_compared += 1
                totals.compare(first.payload, second.payload)
        comparisons[str(threads)] = {**vars(totals), 'passed': totals.passed}
    return rows, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', action='append', type=parse_target, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('data/processed/deepdrid/development_v1.csv'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    targets = sorted(args.target, key=lambda target: target.threads)
    if (tuple(target.threads for target in targets) != THREADS or len({target.container for target in targets}) != 3
            or len({target.url for target in targets}) != 3):
        parser.error('Provide distinct URL/container targets for exactly 1, 2 and 4 threads')
    if args.output.exists():
        parser.error('Measured report exists; preserve it and choose another output')
    images, sample_digest = sample_manifest(args.manifest)
    with ExitStack() as stack:
        clients = {target.threads: stack.enter_context(httpx.Client(base_url=target.url, timeout=120, trust_env=False))
                   for target in targets}
        environment = verify_targets(targets, clients)
        print('Verified the frozen sample and three matching CPU containers. Starting timed requests.', flush=True)
        samples = collect_samples(clients, images)
    rows, equivalence = summarize_samples(samples, targets)
    passed = all(row['passed'] for row in equivalence.values())
    report = {
        'date_utc': datetime.now(timezone.utc).isoformat(), 'status': 'measured' if passed else 'outputs_differ',
        'scope': 'Sequential loopback HTTP timing of a real frozen model on one Docker host',
        'model_version': MODEL_VERSION, 'partition': 'Set-A development validation',
        'sample_rule': 'First twelve frozen selection rows in manifest order; bytes checked by SHA-256',
        'sample_images': PHOTO_COUNT, 'manifest_sha256': MANIFEST_SHA256, 'sample_roster_sha256': sample_digest,
        'benchmark_source_sha256': sha256(Path(__file__).read_bytes()).hexdigest(), 'repeats': REPEATS,
        'warmup_requests_per_target': 2,
        'batch_compute_profile': 'Batch requests of four use one horizontal-flip extra pass',
        'schedule': 'Requests run one at a time, rotating target order per request and workload order per repeat',
        'runtime': environment, 'profiles_by_thread_count': rows,
        'output_equivalence': {'reference_threads': REFERENCE, 'absolute_float_tolerance': ABSOLUTE_TOLERANCE,
                               'excluded_response_fields': sorted(VOLATILE), 'by_thread_count': equivalence},
        'limitations': [
            'Wall time includes multipart upload, JPEG decode, preprocessing, engine, decision and HTTP response.',
            'Client disk reads, response parsing, browser rendering and remote networking are outside the timer.',
            'All other targets were idle during each request. The single host and small sample limit tail-latency inference.',
            'Peak RSS and cgroup memory are high-water marks since container start, including startup and probes.',
            'The timings use development images and do not evaluate clinical performance or concurrent throughput.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'{"Matched" if passed else "Compared with differences"} 1 and 4 threads against 2 using '
          f'{sum(len(rows[str(threads)]["profiles"]) for threads in THREADS)} measured profiles.', flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
