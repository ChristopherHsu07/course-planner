"""Turn one Rice section detail page into a dict the loader can store.

A detail page is the response to ``p_action=COURSE`` for one CRN. The
heading looks like ``ASIA 104 001 (CRN: 26861)``. Labeled facts sit in
``<b>Label:</b> value`` blocks. Meetings are free text such as
``4:00PM - 5:15PM MW (11-JAN-2027 - 23-APR-2027)`` and are expanded into
one row per weekday. Anything the parser does not understand is collected
in ``warnings`` and still returned; only a missing section heading raises.
"""

import re
import warnings
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Rice serves XHTML. Parsing it with the HTML parser is intentional.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Rice day letters. R is Thursday and U is Sunday; there is no "H" or "T" for those.
DAY_NUM = {"M": 1, "T": 2, "W": 3, "R": 4, "F": 5, "S": 6, "U": 7}
# "4:00PM - 5:15PM MW" -> start, end, day letters.
MEETING_RE = re.compile(
    r"(\d{1,2}:\d{2}[AP]M)\s*-\s*(\d{1,2}:\d{2}[AP]M)\s+([A-Z]+)")
# "(11-JAN-2027 - 23-APR-2027)" -> start date, end date.
DATES_RE = re.compile(
    r"\((\d{1,2}-[A-Z]{3}-\d{4})\s*-\s*(\d{1,2}-[A-Z]{3}-\d{4})\)")
# "ASIA 104 001 (CRN: 26861)" -> subject, number, section, crn.
HEADER_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+\(CRN:\s*(\d+)\)")

# Labels the page is known to use. Any other <b> label is kept in the
# field map and also recorded as a warning, so a new Rice field is visible
# in the crawl log without crashing the parse.
KNOWN_LABELS = {
    "Long Title", "Department", "Instructor", "Meeting", "Part of Term",
    "Grade Mode", "Course Type", "Language of Instruction",
    "Distribution Group", "Method of Instruction", "Credit Hours",
    "Course Syllabus", "Course Materials", "Restrictions",
    "Section Max Enrollment", "Section Enrolled", "Waitlisted",
    "Enrollment data as of", "Additional Fees", "Final Exam", "Description",
    "Analyzing Diversity", "Total Cross-list Max Enrollment",
    "Total Cross-list Enrolled", "Instructors",
}


def _date(s):
    """Parse a Rice date like ``11-JAN-2027`` into a ``datetime.date``."""
    return datetime.strptime(s, "%d-%b-%Y").date()


def _time(s):
    """Parse a Rice time like ``4:00PM`` into a ``datetime.time``."""
    return datetime.strptime(s, "%I:%M%p").time()


def _int(s):
    """Return the leading integer in ``s``, or None when there is none.

    Enrollment text is sometimes a bare number and sometimes a note such
    as ``0 (permission required)``. Only the leading digits are the count.
    """
    m = re.match(r"\s*(\d+)", s or "")
    return int(m.group(1)) if m else None


def _fields(soup):
    """Map every bold label on the page to its value string.

    Most facts are ``<div><b>Label: </b>value</div>``: the value is the
    parent's text with the label stripped off. Restrictions puts its value
    in the next sibling ``div``, with its lines joined by ``"; "``. The
    plural Instructors block spreads one person per sibling ``div``, so
    every sibling up to the next labeled field is joined with ``"; "``.
    """
    out = {}
    for b in soup.find_all("b"):
        label = b.get_text(" ", strip=True).rstrip(":").strip()
        if not label:
            continue
        parent = b.parent
        if label == "Restrictions":
            sib = parent.find_next_sibling("div")
            out[label] = "; ".join(sib.stripped_strings) if sib else None
            continue
        if label == "Instructors":
            names = [s for sib in _sibling_divs_until_next_field(parent)
                     for s in sib.stripped_strings]
            out[label] = "; ".join(names) or None
            continue
        text = parent.get_text(" ", strip=True)
        prefix = b.get_text(" ", strip=True)
        value = text[len(prefix):].strip() if text.startswith(prefix) else text
        out[label] = value or None
    return out


def _meeting_text(soup):
    """Return the raw Meeting block, with line breaks preserved.

    Line breaks matter: each schedule line is parsed separately, and a
    section can have more than one (a lecture and a lab, or a date range
    that changes mid-term). Returns ``""`` when the page has no Meeting label.
    """
    for b in soup.find_all("b"):
        label = b.get_text(strip=True)
        if label.startswith("Meeting"):
            return b.parent.get_text("\n", strip=True).replace(label, "", 1).strip()
    return ""


def _parse_meetings(text, default_start, default_end, warns):
    """Expand meeting text into one dict per weekday.

    Each line may carry its own ``(start - end)`` date range. A line
    without one uses ``default_start`` and ``default_end``, which are the
    dates from the first range found anywhere in the block. Day letters
    are split, so ``MW`` becomes Monday and Wednesday at the same times.
    An unknown letter is skipped and appended to ``warns``. A line that
    contains a clock time but does not match the pattern is also a warning,
    so a new layout is noticed instead of dropped silently.
    """
    meetings = []
    for line in text.splitlines():
        dm = DATES_RE.search(line)
        sd, ed = (_date(dm.group(1)), _date(dm.group(2))) if dm \
            else (default_start, default_end)
        found = False
        for m in MEETING_RE.finditer(line):
            found = True
            st, en, days = _time(m.group(1)), _time(m.group(2)), m.group(3)
            for d in days:
                if d not in DAY_NUM:
                    warns.append(f"unknown day letter {d!r} in {line!r}")
                    continue
                meetings.append({"day_of_week": DAY_NUM[d], "start_time": st,
                                 "end_time": en, "start_date": sd,
                                 "end_date": ed})
        if not found and re.search(r"\d:\d\d", line):
            warns.append(f"unparsed meeting line: {line!r}")
    return meetings


def _sibling_divs_until_next_field(div):
    """Yield the ``div`` siblings after ``div`` up to the next labeled field.

    A multi-value block (several instructors) is a label ``div`` followed by
    one ``div`` per value. The next field starts at the first sibling that
    holds a ``<b>`` label (for example ``Meeting:``), so that is where the
    walk stops.
    """
    for sib in div.find_next_siblings("div"):
        if sib.find("b"):
            return
        yield sib


def _instructors(soup):
    """Return ``{"key", "netid", "name"}`` for each instructor on the page.

    Rice uses two layouts, and both are handled:

    * One instructor: ``<div><b>Instructor: </b><a ...>Name</a></div>``.
      The link sits in the same ``div`` as the label.
    * Several instructors: an empty ``<div><b>Instructors: </b></div>``
      followed by one ``<div><a ...>Name</a></div>`` per person.

    A linked name carries ``p_netid`` in the href, and that netid is the
    key. An entry with no link, such as plain-text ``Staff``, is stored with
    ``netid`` None and key ``name:<text>``. Each person becomes its own dict,
    and a person who appears twice is returned once. Empty text never
    produces an instructor.
    """
    out, seen = [], set()

    def add(key, netid, name):
        # De-duplicate by key so one person never gets two link rows.
        if key not in seen:
            seen.add(key)
            out.append({"key": key, "netid": netid, "name": name})

    def collect(div, label=""):
        """Add every instructor found in ``div``; return how many were found."""
        found = 0
        links = div.find_all("a", href=True)
        for a in links:
            qs = parse_qs(urlparse(a["href"]).query)
            netid = (qs.get("p_netid") or [None])[0]
            name = a.get_text(strip=True)
            if name:
                add(netid or f"name:{name}", netid, name)
                found += 1
        if not links:   # e.g. plain-text "Staff"
            txt = div.get_text(" ", strip=True)
            # Drop the label by its real length, not a hardcoded one, so the
            # plural "Instructors:" does not leave a stray ":" behind.
            if label and txt.startswith(label):
                txt = txt[len(label):]
            txt = txt.strip()
            if txt:
                add(f"name:{txt}", None, txt)
                found += 1
        return found

    for b in soup.find_all("b"):
        label = b.get_text(" ", strip=True)
        if not label.startswith("Instructor"):
            continue
        parent = b.parent
        found = collect(parent, label)
        # The plural layout keeps the people in the sibling divs that follow.
        if found == 0 or label.startswith("Instructors"):
            for sib in _sibling_divs_until_next_field(parent):
                collect(sib)
    return out


def parse_detail(html: str) -> dict:
    """Parse a section detail page into the dict ``upsert_section`` expects.

    Raises ``ValueError`` when ``#rice-upflow-data h2`` is not a section
    heading, which is how a form page or an error page is rejected.

    Identity comes from the heading and from ``<body term="...">``.
    ``subject_code`` prefers the ``data-subject-code`` attribute, then the
    heading. Dates are the first range inside the Meeting block. Enrollment
    keeps both the leading integer and the original note; ``permission_required``
    is true when that note mentions permission. Waitlist text like
    ``0 (Max 99)`` splits into the current count and the max. Analyzing
    Diversity is true when its value starts with ``y``. Cross-list counts
    are None when those labels are absent.

    ``warnings`` lists unknown labels, unparsed meeting lines, unknown day
    letters, and an instructor label that yielded no name. The caller logs
    them; they do not fail the parse.
    """
    soup = BeautifulSoup(html, "lxml")
    root = soup.find(id="rice-upflow-data")
    h2 = root.find("h2") if root else None
    hm = HEADER_RE.match(h2.get_text(" ", strip=True)) if h2 else None
    if not hm:
        raise ValueError("not a section detail page")

    warns = []
    f = _fields(soup)
    for label in f:
        if label not in KNOWN_LABELS:
            warns.append(f"unknown field: {label!r} = {f[label]!r}")

    mtext = _meeting_text(soup)
    # Section-level dates: the first range in the whole meeting block.
    dm = DATES_RE.search(mtext)
    start = _date(dm.group(1)) if dm else None
    end = _date(dm.group(2)) if dm else None

    max_txt = f.get("Section Max Enrollment")
    # "3 (Max 99)" -> currently waitlisted, waitlist capacity.
    wl = re.search(r"(\d+)\s*\(Max\s*(\d+)\)", f.get("Waitlisted") or "")
    as_of = f.get("Enrollment data as of")
    h3 = root.find("h3")

    instructors = _instructors(soup)
    if ("Instructor" in f or "Instructors" in f) and not instructors:
        warns.append("instructor label present but no names parsed")
    ad = f.get("Analyzing Diversity")

    return {
        "term_code": soup.body.get("term") if soup.body else None,
        "crn": hm.group(4),
        "subject_code": root.get("data-subject-code") or hm.group(1),
        "course_number": hm.group(2),
        "section_no": hm.group(3),
        "title": h3.get_text(strip=True) if h3 else None,
        "long_title": f.get("Long Title"),
        "description": f.get("Description"),
        "department": f.get("Department"),
        "part_of_term": f.get("Part of Term"),
        "start_date": start, "end_date": end,
        "credits": f.get("Credit Hours"),
        "course_type": f.get("Course Type"),
        "grade_mode": f.get("Grade Mode"),
        "language": f.get("Language of Instruction"),
        "distribution_group": f.get("Distribution Group"),
        "method": f.get("Method of Instruction"),
        "restrictions": f.get("Restrictions"),
        "max_enrollment": _int(max_txt),
        "enrollment_note": max_txt,
        "permission_required": "permission" in (max_txt or "").lower(),
        "enrolled": _int(f.get("Section Enrolled")),
        "waitlisted": int(wl.group(1)) if wl else None,
        "waitlist_max": int(wl.group(2)) if wl else None,
        "enrollment_as_of": (datetime.strptime(as_of, "%d-%b-%Y %I:%M%p")
                             if as_of else None),
        "final_exam": f.get("Final Exam"),
        "meeting_raw": mtext,
        "instructors": instructors,
        "meetings": _parse_meetings(mtext, start, end, warns),
        "warnings": warns,
        "analyzing_diversity": bool(ad and ad.strip().lower().startswith("y")),
        "crosslist_max_enrollment": _int(f.get("Total Cross-list Max Enrollment")),
        "crosslist_enrolled": _int(f.get("Total Cross-list Enrolled")),
    }
