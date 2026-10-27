from __future__ import annotations
import hashlib, json, os, random, tempfile
from pathlib import Path
from typing import Any
import numpy as np
import yaml

def set_seed(seed:int)->None:
    random.seed(seed)
    np.random.seed(seed)

def load_yaml(path:str|Path)->dict:
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r",encoding="utf-8") as f:
        obj=yaml.safe_load(f)
    if not isinstance(obj,dict):
        raise ValueError("Config root must be a mapping")
    return obj

def normalize_text(x:Any)->str:
    return " ".join(str(x or "").strip().split())

def claim_hash(text:str)->str:
    return hashlib.sha256(normalize_text(text).lower().encode("utf-8")).hexdigest()

def first(d:dict,keys:list[str],default=None):
    for k in keys:
        if k in d and d[k] not in (None,""):
            return d[k]
    return default

def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def atomic_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row,ensure_ascii=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise

def atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".tmp",dir=str(path.parent))
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
