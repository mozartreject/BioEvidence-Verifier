"""Conservative input cleanup before evidence retrieval.

Only repair unambiguous, visibly clipped population openers. The submitted
claim remains in the API request and audit record; no verdict words are changed.
"""
import re
import unicodedata


def retrieval_claim(claim):
    clean = ' '.join(unicodedata.normalize('NFKC', claim).split())
    # Common copy/paste clipping: "In adults ..." becomes "n adults ...".
    # Do not apply this to arbitrary tokens or to a claim about the variable n.
    if re.match(r'^n (?:adults|children|patients|people|women|men|infants)\b', clean, re.I):
        return 'I' + clean
    return clean
