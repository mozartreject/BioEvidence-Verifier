import unittest
from types import SimpleNamespace

from dashboard_backend import ImageStatusCapture
from failure_diagnostics import failure_context


class ImageStatusCaptureTests(unittest.TestCase):
    def test_image_status_survives_later_evidence_failure(self):
        adapter = SimpleNamespace(infer=lambda request: SimpleNamespace(
            has_image=True, status=SimpleNamespace(value='SUCCESS')))
        token = failure_context.set({})
        try:
            wrapped = ImageStatusCapture(adapter)
            wrapped.infer(SimpleNamespace(image_path='/tmp/test.png'))
            status = failure_context.get()['image_analysis']
            self.assertEqual(status['status'], 'SUCCESS')
            self.assertTrue(status['has_image'])
            self.assertEqual(status['verdict_effect'], 'none')
        finally:
            failure_context.reset(token)

    def test_text_request_has_no_image_status(self):
        adapter = SimpleNamespace(infer=lambda request: SimpleNamespace(
            has_image=False, status=SimpleNamespace(value='SUCCESS')))
        token = failure_context.set({})
        try:
            ImageStatusCapture(adapter).infer(SimpleNamespace(image_path=None))
            self.assertNotIn('image_analysis', failure_context.get())
        finally:
            failure_context.reset(token)
