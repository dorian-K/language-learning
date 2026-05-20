"""
conftest.py — runs before all other test modules.
Pre-mocks the `requests` and `dotenv` modules so rss.py / scraper.py / config.py
can be imported and tested without those packages being installed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock requests
mock_requests = MagicMock()
mock_requests.exceptions = MagicMock()
mock_requests.exceptions.Timeout = MagicMock()
mock_requests.exceptions.ConnectionError = MagicMock()
sys.modules["requests"] = mock_requests
sys.modules["requests.exceptions"] = mock_requests.exceptions

# Mock dotenv
mock_dotenv = MagicMock()
sys.modules["dotenv"] = mock_dotenv
sys.modules["dotenv"] = MagicMock()
sys.modules["dotenv"].load_dotenv = MagicMock(return_value=None)

# Ensure src/ is on the path for imports like "from bbc_noticias.rss import ..."
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
