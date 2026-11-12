"""The student process the VLN-CE evaluation loop calls (scripts/eval_vlnce.py --agent http).

    HF_HOME=assets/hf_cache env/bin/python -m lightnav.server --ckpt runs/N1/last --port 8765

POST /decide  body: lightnav.vlnce.encode_request(...)  ->  {"answer", "probs", "ms"}
GET  /health  ->  {"ckpt", "params", "device", "tau_stop", "decisions"}
Requests are answered one at a time on one GPU.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from .agent import StudentAgent
from .vlnce import decode_request, encode_response


def make_handler(agent, info):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, json.dumps(info).encode())
            else:
                self._send(404, b'{"error": "unknown path"}')

        def do_POST(self):
            if self.path != "/decide":
                self._send(404, b'{"error": "unknown path"}')
                return
            req = decode_request(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                answer, probs, ms = agent.decide(req["instruction"], req["frames"], req["slots"])
                info["decisions"] += 1
            self._send(200, encode_response(answer, probs, ms))

        def log_message(self, fmt, *args):   # one line per request would flood the log
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help='checkpoint folder (runs/<name>/last), or "laya" for the untrained student')
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tau_stop", type=float, default=None, help="stop only when P(stop) exceeds this")
    args = p.parse_args()
    agent = StudentAgent(args.ckpt, args.device, args.tau_stop)
    info = {"ckpt": args.ckpt, "params": agent.params, "device": args.device, "tau_stop": args.tau_stop, "decisions": 0}
    print("student ready on %s:%d: %s" % (args.host, args.port, json.dumps(info)), flush=True)
    HTTPServer((args.host, args.port), make_handler(agent, info)).serve_forever()


if __name__ == "__main__":
    main()
