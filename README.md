# Rice University Course Planning Assistant (Spring 2027)

This project is an LLM assistant that can answer students' questions about course schedules/planning for the next semester. It runs locally with Docker and Ollama.

## How It Works

1. **Web Scraper** 
The web scraper uses BeautifulSoup to pull course info from [Rice University's Course Catalog](https://courses.rice.edu/courses/!SWKSCAT.cat), parses it, and passes it into the database

2. **Database** 
The core tables are sections (primary key term_code + crn), meetings (one row per weekday), instructors and section_instructors, and subjects. There are also course embeddings to match keywords/semantics from user input. The database is run locally the PostGreSQL on Docker.

3. **Services** 
To help with search, services contains a bunch of tools that querying the database for the agent to use. Most important is search_sections(), which filters by subject, distribution group, level, instructor, open seats, credits, days and time windows. It also uses embeddings to find related keywords.

4. **Agent** 
Ollama local model, defaults to qwen3:8b. Starts by doing a semantic check on the input to extract "busy blocks", then uses the search tools based on what was asked. Since it's a small model, Guard.py ensures the outgoing response actually answers the question.

## How To Run It
Make sure ollama is installed, via `brew install ollama`

Clone the repo, then: 

```bash
docker compose up -d
pip install -r requirements.txt
export DATABASE_URL=postgresql://planner:planner@localhost:5433/rice_planner
python scraper/run.py --subjects COMP,MATH      # or no flag for everything
python -m services.embeddings
ollama pull qwen3:8b && ollama serve
python -m agent
```

## Future Improvements
(Currently, the project is still in development, so it has been tested only on subjects COMP, MATH, ASIA, ENGL, and HIST.)
- Containerizing the full stack, which will require no Ollama installation, and will run fully out of the Docker container
- Improving accuracy for more complex queries, by either upgrading the scale of the LLM or adding more tools for the LLM to use