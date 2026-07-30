# Contributing

Issues and PRs welcome. Setup, test commands, and the config/backend reference live in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Before opening a PR:

```bash
uv run ruff check src tests
uv run pytest -m "not slow" -q
```

Both must pass with no optional ML dependencies installed - the offline suite is the
CI gate.

Ground rules:

- Code comments are dry and technical, in English. No TODO/FIXME placeholders, no
  dead code.
- Commits are imperative and focused: "add scale pyramid to template matcher", one
  concern per commit, no drive-by scope creep.
- New pipeline backends go behind the existing interfaces and get registered in
  `pipeline/factories.py`; heavy dependencies stay importable-optional (see
  docs/DEVELOPMENT.md, "Adding a backend").
- Claims about hardware behavior (optics, power, thermals, WiFi range) need bench
  evidence: state the measurement and how it was taken in the PR description. Listing
  specs and datasheet nominals do not count - this project has been burned by both.
