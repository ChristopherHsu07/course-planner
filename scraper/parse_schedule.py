import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def parse_subjects(form_html: str) -> list[dict]:
    """code -> description, from the Subject Description dropdown."""
    soup = BeautifulSoup(form_html, "lxml")
    sel = soup.find("select", {"name": "p_subj"})
    return [{"code": o["value"], "name": o.get_text(strip=True)}
            for o in sel.find_all("option") if o.get("value")]


def parse_results(html: str) -> list[dict]:
    """One dict per section row on a results page (we only need the CRNs
    to know which detail pages to fetch)."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("tr"):
        crn = tr.select_one("td.cls-crn")
        if crn:
            rows.append({
                "crn": crn.get_text(strip=True),
                "course": tr.select_one("td.cls-crs").get_text(" ", strip=True),
                "title": tr.select_one("td.cls-ttl").get_text(strip=True),
            })
    return rows