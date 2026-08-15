"""Entry point: launch the REST API with uvicorn.

Usage:  python scripts/run_api.py

Then click  http://127.0.0.1:8000  in the terminal (Ctrl+click in VS Code) for
the scoring UI. Other routes: /docs (OpenAPI), /health, /predict (POST).

Ignore the "Uvicorn running on http://0.0.0.0:8000" line uvicorn prints: that
is the *bind* address, not a link. See the HOST comment below.

Press Ctrl+C to stop.
"""

import os
import socket
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, _SRC)
# reload=True re-launches the app in a child process, which does not inherit
# sys.path — so export it too, or the reloader cannot import loan_default.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [_SRC, *filter(None, [os.environ.get("PYTHONPATH")])]
)

import uvicorn  # noqa: E402

# HOST is the address the server BINDS to (listens on), not one you browse to.
#   "0.0.0.0"   -> listen on every network interface this machine has. That is
#                  what makes the app reachable both at 127.0.0.1 (this PC) and
#                  at the LAN IP (phone / another laptop / WSL). Use this when
#                  you want to demo the UI from another device.
#   "127.0.0.1" -> loopback only. Nothing outside this PC can connect. Prefer
#                  this for solo development: it keeps an unauthenticated
#                  scoring endpoint off whatever Wi-Fi you happen to join.
# Either way you OPEN http://127.0.0.1:8000 in the browser -- "0.0.0.0" is a
# wildcard for the listening side and is not a valid destination to type in.
HOST = "0.0.0.0"
PORT = 8000


def _lan_ip() -> str | None:
    """Best-effort LAN address of this machine, for the banner below.

    Opens a UDP socket to a public IP to discover which local interface owns
    the default route. UDP is connectionless, so no packet is actually sent
    and no internet access is required; returns None if there is no route.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return None


if __name__ == "__main__":
    # Printed once by the parent process, before uvicorn's own (bind-address)
    # log line, so the address to actually click is unambiguous.
    print("\n  Loan Default Risk API")
    print("  ---------------------")
    print(f"  UI      ->  http://127.0.0.1:{PORT}/          <-- click this")
    print(f"  Docs    ->  http://127.0.0.1:{PORT}/docs")
    print(f"  Health  ->  http://127.0.0.1:{PORT}/health")
    if HOST == "0.0.0.0":
        lan = _lan_ip()
        if lan:
            # Only reachable while both devices sit on the same network, and
            # only if the Windows firewall permits inbound traffic on PORT.
            print(f"  From another device on this network: http://{lan}:{PORT}/")
    print("  Ctrl+C to stop.\n")

    uvicorn.run(
        "loan_default.service:app",
        host=HOST,
        port=PORT,
        # Dev convenience: watches the source tree and restarts on save. It is
        # why PYTHONPATH is exported above (the reloader runs a child process).
        # Drop reload and add workers=N for a production-style run.
        reload=True,
    )
