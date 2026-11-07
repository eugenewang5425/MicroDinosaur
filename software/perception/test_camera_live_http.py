"""Actual HTTP contract tests with the production state exchange; no camera/NPU."""
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from camera_live_server import make_handler
from preview_runtime import PreviewRuntime


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / 'index.html').write_text('preview', encoding='utf-8')
        self.runtime = PreviewRuntime()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.runtime, self.root / 'index.html', self.root))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = http.client.HTTPConnection(*self.server.server_address, timeout=3)

    def tearDown(self):
        self.runtime.stop()
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.directory.cleanup()

    def request(self, method, path, value=None, headers=None):
        body = json.dumps(value) if value is not None else None
        self.client.request(method, path, body=body, headers=headers or {})
        response = self.client.getresponse()
        return response.status, dict(response.getheaders()), response.read()

    def publish(self):
        model, epoch = self.runtime.model_request()
        self.runtime.activate_model(model, epoch)
        frame = self.runtime.publish_frame(object(), time.monotonic() - .01)
        return self.runtime.publish_result(frame, b'\xff\xd8test\xff\xd9', model, epoch, {'inference_ms': 1.0})

    def test_config_page_and_path_isolation(self):
        self.assertEqual(self.request('GET', '/')[2], b'preview')
        status, _, data = self.request('GET', '/api/config')
        self.assertEqual(status, 200)
        self.assertEqual([item['id'] for item in json.loads(data)['models']], ['midas-optimized-int8'])
        self.assertEqual(self.request('GET', '/results/index.html')[0], 200)
        self.assertEqual(self.request('GET', '/results/%2e%2e/outside.html')[0], 404)

    def test_new_result_cursor_and_current_epoch(self):
        result = self.publish()
        status, headers, data = self.request('GET', '/api/frame?after=0&epoch=1&timeout=0')
        self.assertEqual(status, 200)
        self.assertEqual(headers['X-Model-Id'], 'fast-depth-int8')
        self.assertEqual(headers['X-Model-Epoch'], '1')
        self.assertEqual(int(headers['X-Frame-Id']), result.seq)
        self.assertEqual(data, result.jpeg)
        self.assertEqual(self.request('GET', f'/api/frame?after={result.seq}&epoch=1&timeout=0')[0], 204)
        self.assertEqual(self.request('GET', '/api/frame?after=-1&epoch=1&timeout=0')[0], 400)

    def test_switch_is_async_clears_result_and_rejects_bad_model(self):
        self.publish()
        status, _, body = self.request('POST', '/api/model', {'model': 'midas-optimized-int8'})
        state = json.loads(body)
        self.assertEqual(status, 202)
        self.assertEqual(state['phase'], 'loading')
        self.assertIsNone(state['result'])
        status, headers, _ = self.request('GET', '/api/frame?after=0&epoch=1&timeout=0')
        self.assertEqual(status, 204)
        self.assertEqual(headers['X-Model-Epoch'], '2')
        self.assertEqual(self.request('POST', '/api/model', {'model': '../other'})[0], 400)
        self.assertEqual(self.runtime.model_request(), ('midas-optimized-int8', 2))
        self.assertEqual(self.request('POST', '/api/model', {'model': 'fast-depth-int8'})[0], 400)
        self.assertEqual(self.request('POST', '/api/model', {'model': 'fast-depth-int8'}, {'Origin': 'http://unrelated.test'})[0], 403)

    def test_capture_fault_does_not_serve_previous_result(self):
        self.publish()
        self.runtime.capture_failed('camera lost', 1)
        self.assertEqual(self.request('GET', '/api/frame?after=0&epoch=1&timeout=0')[0], 503)
        status, _, data = self.request('GET', '/api/state')
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(data)['result'])

    def test_stream_preserves_jpeg_and_headers_and_ends_on_epoch_change(self):
        result = self.publish()
        self.client.request('GET', '/api/stream?after=0&epoch=1')
        response = self.client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Transfer-Encoding'), 'chunked')
        self.assertEqual(response.readline(), b'--depthframe\r\n')
        headers = {}
        while True:
            line = response.readline()
            if line == b'\r\n':
                break
            key, value = line.decode('ascii').strip().split(': ', 1)
            headers[key] = value
        self.assertEqual(headers['X-Model-Id'], 'fast-depth-int8')
        self.assertEqual(headers['X-Model-Epoch'], '1')
        self.assertEqual(response.read(int(headers['Content-Length'])), result.jpeg)
        self.runtime.request_model('midas-optimized-int8')
        self.assertEqual(response.read(), b'\r\n')


if __name__ == '__main__':
    unittest.main()
