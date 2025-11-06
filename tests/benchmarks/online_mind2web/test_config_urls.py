# """
# Test that all URLs in the online_mind2web config file are reachable.
# """
# import json
# import time
# from pathlib import Path

# import pytest
# import requests

# CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "src" / "agentlab" / "benchmarks" / "online_mind2web" / "config.json"


# def load_config():
#     """Load the config file and extract all unique URLs."""
#     with open(CONFIG_PATH, "r") as f:
#         config = json.load(f)
    
#     urls = set()
#     for task in config:
#         if "website" in task:
#             urls.add(task["website"])
    
#     return sorted(urls)


# @pytest.fixture(scope="module")
# def unique_urls():
#     """Fixture to load unique URLs from config."""
#     return load_config()


# def test_config_file_exists():
#     """Test that the config file exists."""
#     assert CONFIG_PATH.exists(), f"Config file not found at {CONFIG_PATH}"


# def test_all_tasks_have_website_field():
#     """Test that all tasks have a website field."""
#     with open(CONFIG_PATH, "r") as f:
#         config = json.load(f)
    
#     for i, task in enumerate(config):
#         assert "website" in task, f"Task {i} (id: {task.get('task_id', 'unknown')}) is missing 'website' field"
#         assert task["website"].startswith("http"), f"Task {i} website should be a valid URL starting with http"


# @pytest.mark.parametrize("url", load_config())
# def test_url_is_reachable(url):
#     """
#     Test that each unique URL from the config is reachable.
    
#     This test makes an HTTP HEAD request to each URL to verify it's accessible.
#     We use HEAD instead of GET to minimize bandwidth and load on the servers.
#     """
#     # Configure request timeout and headers
#     timeout = 10  # seconds
#     headers = {
#         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
#     }
    
#     try:
#         # Try HEAD request first (faster, less bandwidth)
#         response = requests.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        
#         # Some servers don't support HEAD, so fall back to GET if needed
#         if response.status_code == 405:  # Method Not Allowed
#             response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        
#         # Check if the response is successful (2xx or 3xx status codes)
#         assert response.status_code < 400, f"URL {url} returned status code {response.status_code}"
        
#     except requests.exceptions.Timeout:
#         pytest.fail(f"URL {url} timed out after {timeout} seconds")
#     except requests.exceptions.ConnectionError as e:
#         pytest.fail(f"Could not connect to URL {url}: {e}")
#     except requests.exceptions.RequestException as e:
#         pytest.fail(f"Request to URL {url} failed: {e}")
    
#     # Add a small delay between requests to be respectful to servers
#     time.sleep(0.5)
