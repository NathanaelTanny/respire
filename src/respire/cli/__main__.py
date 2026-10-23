"""Single entry point for the six commands.

Each command keeps its own module so a reader can open exactly the one they need; this dispatcher
only resolves the name and forwards the remaining arguments.
"""

from __future__ import annotations

import sys

COMMANDS = ("cohort", "train", "evaluate", "compare", "ablate", "certify")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: respire <" + "|".join(COMMANDS) + "> [options]")
        print("       python -m respire.cli.<command> --help")
        return 0 if args else 2
    command = args[0]
    if command not in COMMANDS:
        print(
            f"unknown command {command!r}; expected one of {', '.join(COMMANDS)}", file=sys.stderr
        )
        return 2
    module = __import__(f"respire.cli.{command}", fromlist=["main"])
    return int(module.main(args[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
