"""冷启动复现任务的本地可视化控制台。

服务只监听 127.0.0.1，绝不读取或保存 API 密钥。配置变更写入新的 YAML 文件，
确保冻结的提交配置保持不可变且可审计。
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from src.finqa_agent.runtime.runtime_contract import validate_runtime_contract


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    path: tuple[str, ...]
    kind: str
    minimum: float
    maximum: float
    step: float


FIELDS = (
    Field("temperature", "Temperature", ("model", "temperature"), "float", 0, 2, 0.05),
    Field("top_p", "Top P", ("model", "top_p"), "float", 0, 1, 0.05),
    Field("seed", "随机种子", ("model", "seed"), "int", 0, 2**31 - 1, 1),
    Field(
        "timeout_seconds",
        "API 超时（秒）",
        ("model", "timeout_seconds"),
        "int",
        10,
        1800,
        10,
    ),
    Field("max_retries", "最大重试次数", ("model", "max_retries"), "int", 0, 10, 1),
    Field(
        "retrieval_top_k",
        "稀疏检索 Top K",
        ("profiles", "reproduction", "retrieval_top_k"),
        "int",
        1,
        100,
        1,
    ),
    Field(
        "prompt_pages",
        "Prompt 页数上限",
        ("profiles", "reproduction", "prompt_pages"),
        "int",
        1,
        50,
        1,
    ),
    Field(
        "page_chars",
        "每页字符上限",
        ("profiles", "reproduction", "page_chars"),
        "int",
        100,
        5000,
        50,
    ),
    Field(
        "token_budget",
        "运行 Token 保护预算",
        ("profiles", "reproduction", "token_budget"),
        "int",
        1,
        2_000_000,
        1000,
    ),
    Field(
        "submission_token_limit",
        "提交 Token 上限",
        ("profiles", "reproduction", "submission_token_limit"),
        "int",
        1,
        2_000_000,
        1000,
    ),
)


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key]
    return current


def nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current: Any = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def coerce_updates(raw: dict[str, Any]) -> dict[str, int | float]:
    updates: dict[str, int | float] = {}
    by_key = {field.key: field for field in FIELDS}
    unknown = sorted(set(raw) - set(by_key))
    if unknown:
        raise ValueError(f"unknown fields: {', '.join(unknown)}")
    for key, value in raw.items():
        field = by_key[key]
        number = int(value) if field.kind == "int" else float(value)
        if not field.minimum <= number <= field.maximum:
            raise ValueError(
                f"{field.label} must be between {field.minimum:g} and {field.maximum:g}"
            )
        updates[key] = number
    return updates


def materialize_config(
    source: Path,
    destination: Path,
    raw_updates: dict[str, Any],
) -> dict[str, Any]:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    updates = coerce_updates(raw_updates)
    for field in FIELDS:
        if field.key in updates:
            nested_set(config, field.path, updates[field.key])
    validate_runtime_contract(config, "reproduction", workers=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return config


class DashboardState:
    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.csrf = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.last_command: list[str] = []
        self.last_log = ""

    def snapshot(self) -> dict[str, Any]:
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        with self.lock:
            status = "idle"
            return_code = None
            if self.process is not None:
                return_code = self.process.poll()
                status = "running" if return_code is None else "completed"
        return {
            "csrf": self.csrf,
            "config": str(self.config_path),
            "model": config["model"],
            "fields": [
                {
                    "key": field.key,
                    "label": field.label,
                    "kind": field.kind,
                    "min": field.minimum,
                    "max": field.maximum,
                    "step": field.step,
                    "value": nested_get(config, field.path),
                }
                for field in FIELDS
            ],
            "status": status,
            "return_code": return_code,
            "command": self.last_command,
            "log": self.last_log,
        }

    def launch(
        self,
        *,
        input_path: Path,
        output_path: Path,
        config_path: Path,
        workers: int,
    ) -> list[str]:
        if not input_path.exists():
            raise ValueError(f"input does not exist: {input_path}")
        if output_path.exists() and any(output_path.iterdir()):
            raise ValueError(f"output directory must be empty: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = output_path.parent / f"{output_path.name}.dashboard-launch.log"
        command = [
            sys.executable,
            str(self.root / "run_reproduce.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--workers",
            str(workers),
        ]
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("a reproduction process is already running")
            self.log_handle = log_path.open("w", encoding="utf-8")
            self.process = subprocess.Popen(
                command,
                cwd=self.root,
                stdout=self.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.last_command = command
            self.last_log = str(log_path)
        return command


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Financial-QA 复现控制台</title>
<style>
:root{color-scheme:dark;--bg:#07111f;--panel:#101d2e;--line:#263c55;--accent:#55d6be}
*{box-sizing:border-box}body{margin:0;font:14px system-ui;background:linear-gradient(135deg,#07111f,#10253b);color:#e8f0f7}
main{max-width:1100px;margin:36px auto;padding:0 20px}.hero{display:flex;justify-content:space-between;gap:20px;align-items:end}
h1{font-size:30px;margin:0 0 8px}.tag{color:var(--accent)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:24px}
.card{background:rgba(16,29,46,.96);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 18px 40px #0005}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}label{display:block;color:#a9bfd1;margin-bottom:6px}
input{width:100%;background:#081421;border:1px solid #34506b;border-radius:8px;color:white;padding:10px}
button{border:0;border-radius:8px;padding:11px 16px;font-weight:700;cursor:pointer;background:var(--accent);color:#04211c}
button.secondary{background:#243b52;color:#e8f0f7}.actions{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
pre{white-space:pre-wrap;word-break:break-word;background:#07111f;border-radius:8px;padding:12px;min-height:70px}
.wide{grid-column:1/-1}.status{padding:6px 10px;border:1px solid var(--line);border-radius:999px}
@media(max-width:760px){.grid,.fields{grid-template-columns:1fr}.hero{align-items:start;flex-direction:column}}
</style></head><body><main>
<div class="hero"><div><div class="tag">LOCAL CONTROL PLANE</div><h1>Financial-QA 复现控制台</h1>
<div>配置另存、运行契约校验与一键冷启动复现。API Key 仅从进程环境读取。</div></div>
<div id="status" class="status">加载中</div></div>
<div class="grid">
<section class="card wide"><h2>关键参数</h2><div id="fields" class="fields"></div>
<div class="actions"><button onclick="saveConfig()">另存配置并校验</button>
<button class="secondary" onclick="refresh()">刷新状态</button></div></section>
<section class="card"><h2>运行输入</h2>
<label>性能复现包根目录</label><input id="input" placeholder="D:\\submission\\Financial-QA">
<label style="margin-top:10px">全新输出目录</label><input id="output" placeholder="D:\\reproduce\\run_001">
<label style="margin-top:10px">并发数</label><input id="workers" type="number" value="4" min="1" max="32">
<div class="actions"><button onclick="launch()">确认并启动完整复现</button></div></section>
<section class="card"><h2>审计状态</h2><pre id="message">尚未执行操作。</pre></section>
<section class="card wide"><h2>最近命令</h2><pre id="command"></pre></section>
</div></main>
<script>
let state=null;
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});const x=await r.json();if(!r.ok)throw Error(x.error||r.statusText);return x}
async function refresh(){try{state=await api('/api/state');document.getElementById('status').textContent=state.status+(state.return_code===null?'':' / code '+state.return_code);document.getElementById('command').textContent=(state.command||[]).join(' ')||'尚未启动';if(!document.getElementById('fields').children.length){for(const f of state.fields){const d=document.createElement('div');d.innerHTML=`<label>${f.label}</label><input id="f_${f.key}" type="number" value="${f.value}" min="${f.min}" max="${f.max}" step="${f.step}">`;document.getElementById('fields').appendChild(d)}}}catch(e){msg(e.message)}}
function values(){const x={};for(const f of state.fields)x[f.key]=document.getElementById('f_'+f.key).value;return x}
function msg(x){document.getElementById('message').textContent=x}
async function saveConfig(){try{const x=await api('/api/save',{csrf:state.csrf,values:values()});msg('已保存并通过运行契约校验：\\n'+x.config);await refresh()}catch(e){msg('失败：'+e.message)}}
async function launch(){if(!confirm('将调用 Qwen API 并产生实际费用，确认启动完整复现？'))return;try{const x=await api('/api/launch',{csrf:state.csrf,values:values(),input:document.getElementById('input').value,output:document.getElementById('output').value,workers:document.getElementById('workers').value});msg('已启动：\\n'+x.command.join(' ')+'\\n日志：'+x.log);await refresh()}catch(e){msg('失败：'+e.message)}}
refresh();setInterval(refresh,5000);
</script></body></html>"""


def make_handler(state: DashboardState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/state":
                self._json(state.snapshot())
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if not secrets.compare_digest(str(payload.get("csrf", "")), state.csrf):
                    raise PermissionError("invalid CSRF token")
                configured = state.root / "runtime_index" / "dashboard.reproduction.yaml"
                materialize_config(state.config_path, configured, payload.get("values", {}))
                if self.path == "/api/save":
                    self._json({"config": str(configured)})
                    return
                if self.path == "/api/launch":
                    workers = int(payload.get("workers", 4))
                    if not 1 <= workers <= 32:
                        raise ValueError("workers must be between 1 and 32")
                    command = state.launch(
                        input_path=Path(str(payload.get("input", ""))).resolve(),
                        output_path=Path(str(payload.get("output", ""))).resolve(),
                        config_path=configured,
                        workers=workers,
                    )
                    self._json(
                        {"command": command, "log": state.last_log},
                        HTTPStatus.ACCEPTED,
                    )
                    return
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except PermissionError as exc:
                self._json({"error": str(exc)}, HTTPStatus.FORBIDDEN)
            except Exception as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/config.ultra.yaml"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    state = DashboardState(root, args.config.resolve())
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Financial-QA dashboard: {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
