DROP TABLE IF EXISTS meetings, section_instructors, instructors,
                     sections, course_groups, courses, subjects CASCADE;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE subjects (
  code TEXT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE instructors (
  key   TEXT PRIMARY KEY,        -- netid, or 'name:<name>' if no netid
  netid TEXT,
  name  TEXT NOT NULL
);

CREATE TABLE sections (
  term_code TEXT NOT NULL,
  crn TEXT NOT NULL,
  subject_code TEXT NOT NULL,
  course_number TEXT NOT NULL,
  section_no TEXT NOT NULL,
  title TEXT,
  long_title TEXT,
  description TEXT,
  department TEXT,
  part_of_term TEXT,
  start_date DATE,
  end_date DATE,
  credits TEXT,                  -- text in case of ranges like '1-4'
  course_type TEXT,
  grade_mode TEXT,
  language TEXT,
  distribution_group TEXT,
  method TEXT,
  restrictions TEXT,
  max_enrollment INT,
  enrollment_note TEXT,          -- raw text, e.g. '0 (permission required) ...'
  permission_required BOOLEAN,
  enrolled INT,
  waitlisted INT,
  waitlist_max INT,
  enrollment_as_of TIMESTAMP,
  final_exam TEXT,
  meeting_raw TEXT,
  scraped_at TIMESTAMPTZ DEFAULT now(),
  embedding vector(384),
  PRIMARY KEY (term_code, crn)
);

CREATE TABLE section_instructors (
  term_code TEXT, crn TEXT,
  instructor_key TEXT REFERENCES instructors(key),
  PRIMARY KEY (term_code, crn, instructor_key),
  FOREIGN KEY (term_code, crn) REFERENCES sections
);

CREATE TABLE meetings (
  id SERIAL PRIMARY KEY,
  term_code TEXT NOT NULL, crn TEXT NOT NULL,
  day_of_week SMALLINT NOT NULL,   -- 1=Mon ... 7=Sun
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  start_date DATE, end_date DATE,
  FOREIGN KEY (term_code, crn) REFERENCES sections
);

CREATE INDEX ON sections (subject_code, course_number);
CREATE INDEX ON meetings (term_code, day_of_week, start_time, end_time);

ALTER TABLE sections ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

ALTER TABLE sections ADD COLUMN IF NOT EXISTS analyzing_diversity BOOLEAN;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS crosslist_max_enrollment INT;
ALTER TABLE sections ADD COLUMN IF NOT EXISTS crosslist_enrolled INT;

-- Course page link, e.g. 'http://www.clear.rice.edu/comp140'.
ALTER TABLE sections ADD COLUMN IF NOT EXISTS course_url TEXT;

-- Prerequisites exactly as Rice prints them, e.g.
-- '(MATH 102 OR MATH 106) AND COMP 182'. Kept even when the parse fails.
ALTER TABLE sections ADD COLUMN IF NOT EXISTS prerequisites TEXT;
-- The same expression parsed into JSON: {"and": [...]} / {"or": [...]} with
-- course strings like "COMP 182" as leaves. NULL if absent or unparseable.
ALTER TABLE sections ADD COLUMN IF NOT EXISTS prerequisites_expr JSONB;

-- One row per collapsed "placeholder" course (e.g. COMP 800 GRADUATE RESEARCH).
-- Research sections are one advisor slot each, so dozens share a single
-- course. Every section row is kept; sections.group_id points at its group.
-- Term-independent: the same group serves every term, and the term lives on
-- each section.
CREATE TABLE IF NOT EXISTS course_groups (
  id SERIAL PRIMARY KEY,
  subject_code  TEXT NOT NULL,
  course_number TEXT NOT NULL,
  title         TEXT NOT NULL,
  course_type   TEXT NOT NULL,
  UNIQUE (subject_code, course_number, title)
);
-- NULL for ordinary classes. Set only for course types in COLLAPSE_TYPES
-- (see db/load.py).
ALTER TABLE sections ADD COLUMN IF NOT EXISTS group_id INT REFERENCES course_groups(id);
CREATE INDEX IF NOT EXISTS sections_group_id_idx ON sections (group_id);

-- What browsing, the planner, and future embeddings should read: every
-- ordinary section as its own row, plus one row per group per term. A group
-- row has crn and section_no NULL; section_count says how many CRNs it covers.
-- Look a CRN up in `sections`, and a group's CRNs with
--   SELECT crn FROM sections WHERE group_id = <id>.
CREATE OR REPLACE VIEW catalog_entries AS
SELECT s.term_code, s.crn, s.subject_code, s.course_number, s.section_no,
       s.title, s.course_type, s.credits,
       NULL::int AS group_id, 1 AS section_count
FROM sections s
WHERE s.group_id IS NULL AND s.is_active
UNION ALL
SELECT s.term_code, NULL::text, g.subject_code, g.course_number, NULL::text,
       g.title, g.course_type, min(s.credits),
       g.id, count(*)::int
FROM sections s
JOIN course_groups g ON g.id = s.group_id
WHERE s.is_active
GROUP BY s.term_code, g.id, g.subject_code, g.course_number, g.title,
         g.course_type;