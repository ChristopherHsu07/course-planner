"""Download Rice course-schedule pages and cache the raw HTML.

Every page comes from one Banner endpoint. Callers pass the query
parameters (term, subject, CRN, action) and a cache name. A later crawl
reuses the saved file unless ``force`` is set, so re-running the scraper
does not hit the site again for pages already on disk.
"""

import pathlib
import time

import httpx

# Search form, subject result tables, and section detail pages all live here.
BASE = "https://courses.rice.edu/courses/courses/!SWKSCAT.cat"
ROOT = pathlib.Path(__file__).resolve().parent.parent
# Saved as data/raw/<cache_name>.html. The data/ directory is gitignored.
RAW = ROOT / "data" / "raw"

# Contact address so Rice can reach the operator if the crawl is a problem.
HEADERS = {"User-Agent": "rice-course-planner-prototype (ch212@rice.edu)"}
# Pause after a real network request. A cache hit returns immediately.
DELAY = 1.5


def get(params: dict, cache_name: str, force: bool = False,
        retries: int = 3) -> str:
    """GET one schedule page and return its HTML.

    ``cache_name`` is a path relative to ``data/raw`` without the ``.html``
    suffix, and may include a folder, for example ``202720/detail_26861``.
    An existing file is returned as-is unless ``force`` is true.

    On a cache miss the request is tried up to ``retries`` times. The wait
    between failed attempts grows (3s, then 6s, and so on). After a successful
    download the HTML is written to disk and the caller waits ``DELAY``
    seconds before the next real request. An HTTP error on the last attempt
    is raised.
    """
    path = RAW / f"{cache_name}.html"
    # Cache hit: skip the network and the polite delay.
    if path.exists() and not force:
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            r = httpx.get(BASE, params=params, headers=HEADERS,
                          follow_redirects=True, timeout=30)
            r.raise_for_status()
            break
        except httpx.HTTPError:
            if attempt == retries:
                raise
            # Back off longer on each failure: DELAY * attempt * 2.
            time.sleep(DELAY * attempt * 2)

    path.write_text(r.text, encoding="utf-8")
    time.sleep(DELAY)
    return r.text
