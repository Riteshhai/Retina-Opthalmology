"""Append-only user source testimony bound to immutable IDRiD audit evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from .idrid import IDRID_SOURCE


def idrid_source_confirmation(audit_path: str | Path) -> dict:
    audit_path = Path(audit_path)
    path = audit_path.parent/'source_confirmation.json'
    if not path.is_file():
        return {}
    invalid = {'source_confirmation': {'status': 'invalid_or_stale', 'official_archive_checksum_verified': False}}
    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes)
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if (receipt.get('schema_version') != 1 or receipt.get('dataset') != 'IDRiD' or
                receipt.get('confirmation_kind') != 'user_statement' or
                receipt.get('source_provenance_status') != 'official_download_user_confirmed' or
                not isinstance(receipt.get('confirmation_statement'), str) or
                not receipt['confirmation_statement'].strip() or
                receipt.get('official_archive_checksum_verified') is not False or
                not isinstance(receipt.get('source_urls'), list) or
                IDRID_SOURCE not in receipt.get('source_urls', [])):
            return invalid
        expected = {
            'data_audit_sha256': hashlib.sha256(audit_bytes).hexdigest(),
            'manifest_sha256': audit['private_output_sha256']['external_manifest.csv'],
            'image_content_sha256': audit['image_content_sha256'],
        }
        if any(not re.fullmatch('[0-9a-f]{64}', value) or receipt.get(key) != value
               for key, value in expected.items()):
            return invalid
        timestamp = receipt['confirmed_at_utc']
        parsed = datetime.fromisoformat(timestamp)
        if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
            return invalid
        return {'audit_source_provenance_status': audit.get('source_provenance_status'),
                'source_provenance_status': 'official_download_user_confirmed',
                'source_confirmation': {'status': 'official_origin_user_confirmed',
                    'confirmed_at_utc': timestamp, 'receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
                    'official_archive_checksum_verified': False}}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return invalid
