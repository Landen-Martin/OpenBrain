# OpenBrain

LandenGPT is a Flask and Ollama chat application with streaming responses, tool calls, conversation trash, and runtime settings.

## Layout

- `app.py`: backwards-compatible development entrypoint.
- `src/openbrain/web`: Flask routes, HTML template, CSS, and browser client.
- `src/openbrain/services`: chat streaming orchestration.
- `src/openbrain/domain`: tool implementations and model schemas.
- `db/openbrain`: conversation persistence boundary.
- `arc/architecture`: architecture records and decisions.

## Run

Install the project dependencies used by the original application, ensure Ollama is running with the configured model, then start:

```bash
python app.py
```

Open `http://localhost:5000` in a browser.