from __future__ import annotations

import argparse
from pathlib import Path

from spectator.runtime import checkpoints


def _tail_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return [f"[missing] {path}"]
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump recent trace files and their tails.")
    parser.add_argument("session_id", help="Session identifier to load.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory (default: data).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of recent trace files to show (default: 5).",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=20,
        help="Number of lines to print per trace file (default: 20).",
    )
    args = parser.parse_args()

    checkpoint = checkpoints.load_latest(args.session_id, base_dir=args.base_dir / "checkpoints")
    if checkpoint is None:
        raise SystemExit(f"No checkpoint found for session {args.session_id!r}.")

    trace_dir = args.base_dir / "traces"
    tail = checkpoint.trace_tail[-args.limit :]
    print("Recent traces:")
    for trace_name in tail:
        trace_path = trace_dir / trace_name
        print(f"\n== {trace_name} ==")
        for line in _tail_lines(trace_path, args.lines):
            print(line)


if __name__ == "__main__":
    main()
