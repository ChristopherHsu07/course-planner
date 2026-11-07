"""Database package for the course planner.

``load`` opens Postgres and writes subjects, sections, meetings, and
instructors parsed from the Rice schedule. The schema itself lives in
``schema.sql`` and is applied by Docker on a fresh data volume.
"""
