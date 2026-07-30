"""Exception hierarchy shared across the server."""


class LstkError(Exception):
    """Base class for all lstk-eye errors."""


class ConfigError(LstkError):
    """Invalid or missing configuration."""


class DependencyError(LstkError):
    """An optional dependency is not installed.

    Carries an install hint so CLI/doctor output can tell the user exactly
    what to run.
    """

    def __init__(self, message: str, install_hint: str | None = None):
        super().__init__(message)
        self.install_hint = install_hint


class PipelineError(LstkError):
    """A pipeline stage failed (STT, segmentation, planning, relocalization)."""


class PlanningError(PipelineError):
    """The planner returned an unusable plan (invalid marks, empty steps, API failure)."""
