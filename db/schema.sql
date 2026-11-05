CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE courses (
  id SERIAL PRIMARY KEY,
  subject_code TEXT NOT NULL,          -- 'ASIA'
  number TEXT NOT NULL,                -- '101' (text: 'COMP 140X', etc.)
  title TEXT,
  description TEXT,
  prereq_text TEXT,                    -- always keep raw
  prereq_parsed JSONB,                 -- optional structured version
  embedding vector(384),
  UNIQUE (subject_code, number)
);

CREATE TABLE sections (
  term_code TEXT NOT NULL,             -- '202720'
  crn TEXT NOT NULL,                   -- CRNs are only unique per term
  course_id INT REFERENCES courses(id),
  section_no TEXT,
  part_of_term TEXT,                   -- matters for half-term conflicts
  method TEXT,
  distribution_group TEXT,
  credits TEXT,
  scraped_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (term_code, crn)
);

CREATE TABLE meetings (
  term_code TEXT, crn TEXT,
  day_of_week SMALLINT,                -- 1=Mon ... 7=Sun
  start_time TIME, end_time TIME,
  FOREIGN KEY (term_code, crn) REFERENCES sections
);