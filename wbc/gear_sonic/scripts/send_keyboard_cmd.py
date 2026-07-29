"""Send keyboard commands to ZMQ subscribers on the Sonic keyboard channel.

Examples:
    python gear_sonic/scripts/send_keyboard_cmd.py i
    python gear_sonic/scripts/send_keyboard_cmd.py k i p --interval-sec 0.5
"""

from __future__ import annotations

import argparse
import time

import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish keyboard command(s) to ZMQ, e.g. i/k/p."
    )
    parser.add_argument(
        "keys",
        nargs="+",
        help="Key(s) to send, e.g. i k p.",
    )
    parser.add_argument(
        "--bind-host",
        type=str,
        default="*",
        help="Host for PUB socket bind (default: *).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5580,
        help="ZMQ port (default: 5580).",
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after bind before sending first key (default: 1.0).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=1.0,
        help="Seconds between keys (default: 1.0).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the full key sequence N times (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.warmup_sec < 0 or args.interval_sec < 0:
        raise ValueError("--warmup-sec and --interval-sec must be >= 0")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    endpoint = f"tcp://{args.bind_host}:{args.port}"
    sock.bind(endpoint)
    print(f"[send_keyboard_cmd] Bound PUB at {endpoint}")

    try:
        if args.warmup_sec > 0:
            time.sleep(args.warmup_sec)

        for _ in range(args.repeat):
            for key in args.keys:
                print("send", key)
                sock.send_string(key)
                if args.interval_sec > 0:
                    time.sleep(args.interval_sec)
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
