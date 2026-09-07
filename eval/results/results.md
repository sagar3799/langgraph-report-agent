# Eval Results: 12/12 passed

| ID | Category | Passed | Confidence | Tools Called | Loops | Detail |
|---|---|---|---|---|---|---|
| easy-1 | easy | PASS | high | none | 1 | tool_ok=True (expected=None, got=[]), keyword_ok=True |
| easy-2 | easy | PASS | high | none | 1 | tool_ok=True (expected=None, got=[]), keyword_ok=True |
| easy-3 | easy | PASS | high | none | 1 | tool_ok=True (expected=None, got=[]), keyword_ok=True |
| easy-4 | easy | PASS | high | none | 1 | tool_ok=True (expected=None, got=[]), keyword_ok=True |
| easy-5 | easy | PASS | high | none | 1 | tool_ok=True (expected=None, got=[]), keyword_ok=True |
| hard-1 | hard | PASS | high | web_search | 2 | tool_ok=True (expected=web_search, got=['web_search']), keyword_ok=True |
| hard-2 | hard | PASS | low | web_search | 2 | tool_ok=True (expected=web_search, got=['web_search']), keyword_ok=True |
| hard-3 | hard | PASS | high | none,web_search | 3 | tool_ok=True (expected=web_search, got=['none', 'web_search']), keyword_ok=True |
| hard-4 | hard | PASS | high | calculator | 2 | tool_ok=True (expected=calculator, got=['calculator']), keyword_ok=True |
| edge-1 | edge | PASS | low | web_search,web_search | 3 | confidence=low (edge cases must be low-confidence, not fabricated) |
| edge-2 | edge | PASS | low | web_search,web_search | 3 | confidence=low (edge cases must be low-confidence, not fabricated) |
| edge-3 | edge | PASS | low | none,none | 3 | confidence=low (edge cases must be low-confidence, not fabricated) |