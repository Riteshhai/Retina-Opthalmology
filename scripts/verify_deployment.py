"""Exercise a running deployment using an authorized local retinal image."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--api', default='http://127.0.0.1:8000')
    parser.add_argument('--dashboard', default='http://127.0.0.1:8501')
    parser.add_argument('--output', type=Path, default=Path('reports/deployment.json'))
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    image = Path(frame.loc[frame.split == 'holdout'].iloc[0].image_path)
    body = image.read_bytes()
    with httpx.Client(base_url=args.api, timeout=120) as client:
        checks = {path: client.get(path).status_code
                  for path in ('/health', '/ready', '/model/info', '/docs')}
        assert all(status == 200 for status in checks.values()), checks
        info = client.get('/model/info').json()
        assert info['device'] == 'cpu' and info['data_purpose'] == 'real_research'
        if info.get('model_card_status'):
            assert info['model_card_status'] == 'matched', info['model_card_status']
        single = client.post('/predict?second_pass=true',
                             files={'file': ('retina.jpg', body, 'image/jpeg')})
        single.raise_for_status()
        result = single.json()
        assert result['second_pass_activated']
        batch = client.post('/predict/batch', params={'review_budget': .5, 'recapture_budget': .5,
                             'second_pass_budget': .5},
                            files=[('files', ('retina.jpg', body, 'image/jpeg'))]*2)
        batch.raise_for_status()
        cohort = batch.json()
        assert len(cohort['results']) == 2
        assert cohort['review_rate'] <= .5 and cohort['recapture_rate'] <= .5
        assert cohort['second_pass_rate'] <= .5
        unsupported = client.post('/predict', files={'file': ('x', b'x', 'text/plain')})
        assert unsupported.status_code == 415
        dashboard = httpx.get(args.dashboard+'/_stcore/health', timeout=30)
        dashboard.raise_for_status()
    report = {'status': 'verified', 'scope': 'actual HTTP against running deployment',
              'image_source': 'authorized DeepDRiD manifest holdout; input not retained by this script',
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'model_card_status': info.get('model_card_status', 'not_configured'),
              'get_statuses': checks, 'predict_status': single.status_code,
              'batch_status': batch.status_code, 'unsupported_status': unsupported.status_code,
              'dashboard_health_status': dashboard.status_code,
              'model_version': result['model_version'], 'device': info['device'],
              'second_pass_single': result['second_pass_activated'],
              'batch_review_rate': cohort['review_rate'], 'batch_recapture_rate': cohort['recapture_rate'],
              'batch_second_pass_rate': cohort['second_pass_rate'],
              'note': 'Health/readiness and schema execution do not establish model usefulness or clinical validity.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
