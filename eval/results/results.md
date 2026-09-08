# Eval Results: 11/12 passed

| ID | Category | Passed | Confidence | Tools Called | Loops | Detail |
|---|---|---|---|---|---|---|
| easy-1 | easy | PASS | high | none | 1 | tool_ok=True (want=None, got=[]), kw_ok=True |
| easy-2 | easy | PASS | high | none | 1 | tool_ok=True (want=None, got=[]), kw_ok=True |
| easy-3 | easy | PASS | high | none | 1 | tool_ok=True (want=None, got=[]), kw_ok=True |
| easy-4 | easy | PASS | high | none | 1 | tool_ok=True (want=None, got=[]), kw_ok=True |
| easy-5 | easy | PASS | high | none | 1 | tool_ok=True (want=None, got=[]), kw_ok=True |
| hard-1 | hard | PASS | low | web_search | 2 | tool_ok=True (want=web_search, got=['web_search']), kw_ok=True |
| hard-2 | hard | PASS | high | web_search | 2 | tool_ok=True (want=web_search, got=['web_search']), kw_ok=True |
| hard-3 | hard | PASS | high | web_search | 2 | tool_ok=True (want=web_search, got=['web_search']), kw_ok=True |
| hard-4 | hard | FAIL | high | none | 1 | tool_ok=False (want=calculator, got=[]), kw_ok=True |
| edge-1 | edge | PASS | low | web_search,web_search | 3 | confidence=low (edge cases must be low-confidence) |
| edge-2 | edge | PASS | low | web_search,web_search | 3 | confidence=low (edge cases must be low-confidence) |
| edge-3 | edge | PASS | low | web_search,web_search | 3 | confidence=low (edge cases must be low-confidence) |