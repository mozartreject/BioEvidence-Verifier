"""Local-only preview with a fixed backend bridge. Not a Netlify server."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPConnection
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def bridge(self):
        if self.headers.get('Host') not in ('127.0.0.1:3001', 'localhost:3001'):
            self.send_error(403)
            return
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://127.0.0.1:3001', 'http://localhost:3001'):
            self.send_error(403)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.send_error(400)
            return
        if size < 0 or size > 16 * 1024 * 1024:
            self.send_error(413)
            return
        conn = HTTPConnection('127.0.0.1', 8000, timeout=300)
        try:
            body = self.rfile.read(size) if size else None
            headers = {'Accept': 'application/json'}
            if self.headers.get('Content-Type'):
                headers['Content-Type'] = self.headers['Content-Type']
            conn.request(self.command, self.path, body=body, headers=headers)
            response = conn.getresponse()
            content = response.read()
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(content)
        except (OSError, TimeoutError):
            content = json.dumps({'detail': 'Cannot reach the local backend on port 8000. Check its terminal.'}).encode()
            try:
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except OSError:
                pass
        finally:
            conn.close()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.startswith('/api/v1/'):
            self.bridge()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/v1/inferences':
            self.bridge()
        else:
            self.send_error(404)

if __name__ == '__main__':
    print('Dashboard: http://127.0.0.1:3001', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 3001), Handler).serve_forever()
