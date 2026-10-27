"""Start the local research dashboard; never terminate pre-existing servers."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / 'dashboard-runtime' / 'launcher-logs'
children = []
handles = []


def get(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return response.read()


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def start(name, args):
    LOGS.mkdir(parents=True, exist_ok=True)
    handle = (LOGS / (name + '.log')).open('a')
    handles.append(handle)
    child = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT)
    children.append((name, child))
    print('Starting {}. Log: {}'.format(name, handle.name), flush=True)


def wait_for(test, seconds, description):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for name, child in children:
            if child.poll() is not None:
                raise RuntimeError('{} exited; see {}'.format(name, LOGS / (name + '.log')))
        try:
            if test():
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError('Timed out waiting for ' + description)


def backend_ready():
    data = json.loads(get('http://127.0.0.1:8000/api/v1/system/health'))
    if data.get('verifier_backend') != 'passage_combiner':
        raise RuntimeError('Port 8000 belongs to a different backend. Stop that server in its terminal, then rerun this launcher.')
    if not data.get('retrieval', {}).get('live_pubmed_enabled'):
        raise RuntimeError('Port 8000 is running the older local-only backend. Stop that server in its terminal, then rerun this launcher.')
    return data.get('pipeline_ready') is True


def frontend_ready():
    page = get('http://127.0.0.1:3001/').decode()
    if 'BiomedicalVerifier' not in page or 'claim-form' not in page:
        raise RuntimeError('Port 3001 belongs to a different frontend. Stop that server in its terminal, then rerun this launcher.')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Check existing services without starting or opening anything')
    args = parser.parse_args()
    if args.check:
        assert backend_ready(), 'Backend is still loading'
        assert frontend_ready(), 'Frontend is unavailable'
        print('Improved backend and frontend are ready.')
        return
    if not listening(11434):
        ollama = shutil.which('ollama')
        if not ollama:
            raise RuntimeError('Ollama is missing. Install/open Ollama and rerun this command.')
        start('ollama', [ollama, 'serve'])
    wait_for(lambda: get('http://127.0.0.1:11434/api/tags'), 60, 'Ollama')
    models = json.loads(get('http://127.0.0.1:11434/api/tags')).get('models', [])
    if not any(m.get('name') == 'qwen2.5:7b' for m in models):
        raise RuntimeError('Required model missing. Run: ollama pull qwen2.5:7b')
    if not listening(8000):
        start('backend', ['zsh', str(ROOT / 'start_backend.command')])
    print('Waiting for the improved backend to load its models…', flush=True)
    wait_for(backend_ready, 600, 'backend model loading')
    if not listening(3001):
        start('frontend', ['zsh', str(ROOT / 'start_frontend.command')])
    wait_for(frontend_ready, 30, 'frontend')
    print('Ready: http://127.0.0.1:3001/\nKeep this terminal open. Ctrl+C stops only services started by this launcher.', flush=True)
    subprocess.run(['open', 'http://127.0.0.1:3001/'], check=False)
    while children:
        for name, child in children:
            if child.poll() is not None:
                raise RuntimeError(name + ' stopped; check its launcher log.')
        time.sleep(2)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopping services started by this launcher.')
    except Exception as exc:
        print('Launcher error: {}'.format(exc), flush=True)
        raise SystemExit(1)
    finally:
        for _, child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()
