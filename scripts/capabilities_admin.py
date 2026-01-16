from __future__ import annotations

import argparse
import sys

from spectator.runtime.capabilities import (
    clear_pending,
    grant_permission,
    revoke_permission,
)
from spectator.runtime.checkpoints import load_latest, save_checkpoint


def _validate_capability(value: str | None) -> str:
    if value is None:
        raise ValueError("capability is required")
    cap = value.strip()
    if not cap:
        raise ValueError("capability must be a non-empty string")
    return cap


def _load_checkpoint(session_id: str):
    checkpoint = load_latest(session_id)
    if checkpoint is None:
        raise ValueError(f"session {session_id!r} not found")
    return checkpoint


def _handle_list(args: argparse.Namespace) -> int:
    checkpoint = _load_checkpoint(args.session)
    print("granted:")
    for cap in checkpoint.state.capabilities_granted:
        print(f"- {cap}")
    print("pending:")
    for cap in checkpoint.state.capabilities_pending:
        print(f"- {cap}")
    return 0


def _handle_grant(args: argparse.Namespace) -> int:
    checkpoint = _load_checkpoint(args.session)
    cap = _validate_capability(args.cap)
    if grant_permission(checkpoint.state, cap):
        save_checkpoint(checkpoint)
    return 0


def _handle_revoke(args: argparse.Namespace) -> int:
    checkpoint = _load_checkpoint(args.session)
    cap = _validate_capability(args.cap)
    if revoke_permission(checkpoint.state, cap):
        save_checkpoint(checkpoint)
    return 0


def _handle_clear_pending(args: argparse.Namespace) -> int:
    checkpoint = _load_checkpoint(args.session)
    if clear_pending(checkpoint.state):
        save_checkpoint(checkpoint)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage session capabilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List granted and pending capabilities.")
    list_parser.add_argument("--session", required=True, help="Session id")
    list_parser.set_defaults(handler=_handle_list)

    grant_parser = subparsers.add_parser("grant", help="Grant a capability.")
    grant_parser.add_argument("--session", required=True, help="Session id")
    grant_parser.add_argument("--cap", required=True, help="Capability string")
    grant_parser.set_defaults(handler=_handle_grant)

    revoke_parser = subparsers.add_parser("revoke", help="Revoke a capability.")
    revoke_parser.add_argument("--session", required=True, help="Session id")
    revoke_parser.add_argument("--cap", required=True, help="Capability string")
    revoke_parser.set_defaults(handler=_handle_revoke)

    clear_parser = subparsers.add_parser("clear-pending", help="Clear pending capabilities.")
    clear_parser.add_argument("--session", required=True, help="Session id")
    clear_parser.set_defaults(handler=_handle_clear_pending)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
