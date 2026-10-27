from __future__ import annotations
import json,time
import requests

class OllamaClient:
    def __init__(self,cfg:dict):
        self.url=cfg["ollama_url"]
        self.model=cfg["model"]
        self.cfg=cfg

    def chat(self,system_prompt:str,user_prompt:str):
        payload={
            "model":self.model,
            "stream":False,
            "format":"json",
            "options":{
                "temperature":float(self.cfg["temperature"]),
                "top_p":float(self.cfg["top_p"]),
                "num_predict":int(self.cfg["num_predict"])
            },
            "messages":[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_prompt}
            ]
        }
        last=None
        for attempt in range(int(self.cfg["request_retries"])):
            try:
                response=requests.post(self.url,json=payload,timeout=int(self.cfg["timeout_seconds"]))
                response.raise_for_status()
                body=response.json()
                content=body.get("message",{}).get("content")
                if not isinstance(content,str) or not content.strip():
                    raise RuntimeError("Ollama response missing message.content")
                output=json.loads(content)
                if not isinstance(output,dict):
                    raise RuntimeError("Expected JSON object from generator")
                meta={
                    "model":body.get("model",self.model),
                    "done_reason":body.get("done_reason"),
                    "prompt_eval_count":body.get("prompt_eval_count"),
                    "eval_count":body.get("eval_count")
                }
                return output,meta
            except Exception as e:
                last=e
                if attempt+1<int(self.cfg["request_retries"]):
                    time.sleep(min(2**attempt,8))
        raise RuntimeError(f"Ollama failed after retries: {last}")
