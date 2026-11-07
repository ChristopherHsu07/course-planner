"""Parse the search form and a subject results page.

The form page supplies the subject catalog. A results page is the table
Rice returns for one subject in one term; each row is one section, and
the crawler uses the CRN to decide which detail page to fetch next.
"""

import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Rice serves XHTML. Parsing it with the HTML parser is intentional.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


def parse_subjects(form_html: str) -> list[dict]:
    """Return ``{"code", "name"}`` for every subject offered this term.

    Reads the Subject Description dropdown (``<select name="p_subj">``),
    not the separate Subject Code dropdown. Options with an empty value,
    the blank "any subject" choice, are skipped. ``code`` is the option
    value (``ASIA``); ``name`` is the visible label (``Asian Studies``).
    """
    soup = BeautifulSoup(form_html, "lxml")
    sel = soup.find("select", {"name": "p_subj"})
    return [{"code": o["value"], "name": o.get_text(strip=True)}
            for o in sel.find_all("option") if o.get("value")]


def parse_results(html: str) -> list[dict]:
    """Return one dict per section row on a subject results page.

    A row counts only when it has a ``td.cls-crn``. The crawler needs
    ``crn`` to request the detail page. ``course`` (for example
    ``ASIA 104 001``) and ``title`` are kept so a later failure can be
    logged with a readable name. Rows that are headers or padding are
    ignored because they have no CRN cell.
    """
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
