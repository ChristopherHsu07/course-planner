DROP TABLE IF EXISTS meetings, section_instructors, instructors,
                     sections, courses, subjects CASCADE;
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