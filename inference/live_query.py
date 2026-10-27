"""General secondary PubMed query for long biomedical assertions.

Keeps the first content term and the final two content terms. The primary
search still uses the complete filtered claim; this query broadens recall.
The claim supplied to the verifier is never rewritten.
"""
import re


def compact_terms(primary_query):
    tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", primary_query.lower())
    filler = {"against", "among", "between", "people", "person", "persons",
              "patients", "individuals", "individual", "never", "always", "often"}
    content = [token for token in tokens if token not in filler]
    if len(content) <= 3:
        result = " ".join(dict.fromkeys(content))
        return result if result != " ".join(tokens) else ""
    compact = list(dict.fromkeys([content[0], *content[-2:]]))
    result = " ".join(compact)
    return result if result != " ".join(content) else ""
