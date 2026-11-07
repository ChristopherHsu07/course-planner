import pathlib
import time

import httpx

BASE = "https://courses.rice.edu/courses/courses/!SWKSCAT.cat"
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# Put a real contact address here so Rice can reach you if needed
HEADERS = {"User-Agent": "rice-course-planner-prototype (ch212@rice.edu)"}
DELAY = 1.5  # seconds between real requests


def get(params: dict, cache_name: str, force: bool = False,
        retries: int = 3) -> str:
    """GET a page, caching raw HTML on disk. cache_name may include a
    folder, e.g. '202720/detail_26861'."""
    path = RAW / f"{cache_name}.html"
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
            time.sleep(DELAY * attempt * 2)

    path.write_text(r.text, encoding="utf-8")
    time.sleep(DELAY)
    return r.text