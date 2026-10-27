import unittest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from dashboard_backend import create_app
class ErrorContractTests(unittest.TestCase):
 def test_evidence_failure_is_explicit_422(self):
  app=create_app()
  @app.get('/_test/evidence-contract')
  def fail():
   from failure_diagnostics import failure_context
   failure_context.get()['diagnostics']={'claim':'Request-local test','accepted_verdict':None}
   raise HTTPException(500,detail={'error':'INFERENCE_FAILED','message':'Evidence validation failed: no passages justify the result.'})
  response=TestClient(app).get('/_test/evidence-contract')
  self.assertEqual(response.status_code,422)
  self.assertEqual(response.json()['detail']['error'],'EVIDENCE_VALIDATION_FAILED')
  self.assertNotIn('final_verdict',response.json())
  self.assertEqual(response.json()['detail']['diagnostics']['claim'],'Request-local test')
 def test_other_errors_preserved(self):
  app=create_app()
  @app.get('/_test/unrelated-contract')
  def fail():raise HTTPException(503,detail='Model not loaded')
  response=TestClient(app).get('/_test/unrelated-contract')
  self.assertEqual(response.status_code,503);self.assertEqual(response.json()['detail'],'Model not loaded')
if __name__=='__main__':unittest.main()
