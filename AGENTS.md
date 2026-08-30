# Agent instructions

## Project

Interactive Streamlit app comparing 7,000+ LLMs across 200+ providers on a 3D
chart of cost, speed, and intelligence. Data is fetched live from the
models.dev API (keyless) and optionally the Artificial Analysis API (free key).

## Commit convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short imperative summary>
```

Types:
- `feat:` — new feature or capability
- `fix:` — bug fix
- `docs:` — documentation or README changes
- `refactor:` — code change that does not alter behavior
- `chore:` — maintenance, tooling, dependencies
- `test:` — adding or updating tests

Rules:
- Summary is lowercase, imperative, under ~72 chars.
- Commit to `main` and push after each change: `git commit -m "<type>: <summary>" && git push`.

## Run & verify

```bash
.venv/bin/streamlit run app.py        # start the app (http://localhost:8501)
.venv/bin/python -c "import ast; ast.parse(open('app.py').read())"   # syntax check
```

Before committing, restart the app and confirm it serves HTTP 200:
`curl -s -o /dev/null -w "%{http_code}" http://localhost:8501`

## Notes

- `.app_state.json` persists the sidebar UI config and is gitignored — never commit it.
- The Artificial Analysis API key is stored in `~/.config/model-compare/aa_key`.
