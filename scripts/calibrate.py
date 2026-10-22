"""Fit temperatures only on a dedicated calibration split."""
from pathlib import Path
import sys
import argparse
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.calibration.temperature import write_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr-checkpoint", required=True)
    parser.add_argument("--quality-checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--allow-software-test", action="store_true")
    args = parser.parse_args()
    artifact = write_calibration(args.dr_checkpoint, args.quality_checkpoint, args.manifest, args.output,
                                 allow_software_test=args.allow_software_test,
                                 batch_size=args.batch_size, cpu_threads=args.cpu_threads)
    print(json.dumps({"output": args.output, "data_purpose": artifact["data_purpose"],
                      "dr_temperature": artifact["dr"]["heads"]["dr_grade"]["temperature"],
                      "note": artifact["note"]}, indent=2))


if __name__ == "__main__":
    main()
