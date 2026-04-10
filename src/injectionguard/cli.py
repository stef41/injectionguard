"""Command-line interface for injectionguard."""

from __future__ import annotations

import argparse
import json
import sys

from injectionguard import __version__
from injectionguard.detector import Detector, ThreatLevel

_THRESHOLD_MAP = {
    "low": ThreatLevel.LOW,
    "medium": ThreatLevel.MEDIUM,
    "high": ThreatLevel.HIGH,
    "critical": ThreatLevel.CRITICAL,
}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="injectionguard",
        description="Prompt injection detection for LLM applications and MCP servers",
    )
    parser.add_argument("--version", action="version", version=f"injectionguard {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    scan_p = subparsers.add_parser("scan", help="Scan text for prompt injections")
    scan_p.add_argument("text", nargs="?", help="Text to scan (or use stdin)")
    scan_p.add_argument("--file", "-f", help="Read from file")
    scan_p.add_argument("--threshold", choices=list(_THRESHOLD_MAP), default="low")
    scan_p.add_argument("--format", choices=["text", "json"], default="text")

    batch_p = subparsers.add_parser("batch", help="Scan JSONL file")
    batch_p.add_argument("file", help="JSONL file to scan")
    batch_p.add_argument("--field", default="text", help="JSON field with text")
    batch_p.add_argument("--threshold", choices=list(_THRESHOLD_MAP), default="low")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "scan":
        return _cmd_scan(args)
    elif args.command == "batch":
        return _cmd_batch(args)

    return 0


def _cmd_scan(args) -> int:
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    detector = Detector(threshold=_THRESHOLD_MAP[args.threshold])
    result = detector.scan(text)

    if args.format == "json":
        data = {
            "is_safe": result.is_safe,
            "threat_level": result.threat_level.value,
            "detections": [
                {"strategy": d.strategy, "threat_level": d.threat_level.value,
                 "message": d.message, "offset": d.offset}
                for d in result.detections
            ],
        }
        print(json.dumps(data, indent=2))
    else:
        print(result)

    return 0 if result.is_safe else 1


def _cmd_batch(args) -> int:
    detector = Detector(threshold=_THRESHOLD_MAP[args.threshold])
    total = 0
    flagged = 0

    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get(args.field, "")
            except json.JSONDecodeError:
                text = line

            total += 1
            result = detector.scan(text)
            if not result.is_safe:
                flagged += 1
                print(f"Line {total}: {result.threat_level.value} - {len(result.detections)} detection(s)")

    print(f"\n{total} texts scanned, {flagged} flagged")
    return 1 if flagged > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
