import re
import warnings
from datetime import datetime
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DAY_NUM = {"M": 1, "T": 2, "W": 3, "R": 4, "F": 5, "S": 6, "U": 7}
MEETING_RE = re.compile(
    r"(\d{1,2}:\d{2}[AP]M)\s*-\s*(\d{1,2}:\d{2}[AP]M)\s+([A-Z]+)")
DATES_RE = re.compile(
    r"\((\d{1,2}-[A-Z]{3}-\d{4})\s*-\s*(\d{1,2}-[A-Z]{3}-\d{4})\)")
HEADER_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+\(CRN:\s*(\d+)\)")

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
    return datetime.strptime(s, "%d-%b-%Y").date()


def _time(s):
    return datetime.strptime(s, "%I:%M%p").time()


def _int(s):
    m = re.match(r"\s*(\d+)", s or "")
    return int(m.group(1)) if m else None


def _fields(soup):
    """Every <div><b>Label: </b>value</div> pair on the page."""
    out = {}
    for b in soup.find_all("b"):
        label = b.get_text(" ", strip=True).rstrip(":").strip()
        if not label:
            continue
        parent = b.parent
        if label in ("Restrictions", "Instructors"):
            sib = parent.find_next_sibling("div")
            out[label] = "; ".join(sib.stripped_strings) if sib else None
            continue
        text = parent.get_text(" ", strip=True)
        prefix = b.get_text(" ", strip=True)
        value = text[len(prefix):].strip() if text.startswith(prefix) else text
        out[label] = value or None
    return out


def _meeting_text(soup):
    for b in soup.find_all("b"):
        label = b.get_text(strip=True)
        if label.startswith("Meeting"):
            return b.parent.get_text("\n", strip=True).replace(label, "", 1).strip()
    return ""


def _parse_meetings(text, default_start, default_end, warns):
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


def _instructors(soup):
    out = []
    for b in soup.find_all("b"):
        if b.get_text(strip=True).startswith("Instructor"):
            for a in b.parent.find_all("a", href=True):
                qs = parse_qs(urlparse(a["href"]).query)
                netid = (qs.get("p_netid") or [None])[0]
                name = a.get_text(strip=True)
                out.append({"key": netid or f"name:{name}",
                            "netid": netid, "name": name})
            if not out:   # e.g. plain-text "Staff"
                txt = b.parent.get_text(" ", strip=True)[len("Instructor:"):].strip()
                if txt:
                    out.append({"key": f"name:{txt}", "netid": None, "name": txt})
    return out


def parse_detail(html: str) -> dict:
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
    dm = DATES_RE.search(mtext)
    start = _date(dm.group(1)) if dm else None
    end = _date(dm.group(2)) if dm else None

    max_txt = f.get("Section Max Enrollment")
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