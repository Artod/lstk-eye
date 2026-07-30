"""Tests for lstk_eye.intents voice-command matching."""

import pytest

from lstk_eye.intents import match_intent

KEYWORD_CASES = [
    ("next", "next"),
    ("forward", "next"),
    ("дальше", "next"),
    ("далее", "next"),
    ("вперед", "next"),
    ("вперёд", "next"),
    ("back", "prev"),
    ("previous", "prev"),
    ("назад", "prev"),
    ("repeat", "repeat"),
    ("again", "repeat"),
    ("повтори", "repeat"),
    ("повторить", "repeat"),
    ("еще раз", "repeat"),
    ("ещё раз", "repeat"),
    ("cancel", "cancel"),
    ("stop", "cancel"),
    ("done", "cancel"),
    ("exit", "cancel"),
    ("quit", "cancel"),
    ("отмена", "cancel"),
    ("стоп", "cancel"),
    ("хватит", "cancel"),
    ("готово", "cancel"),
]


@pytest.mark.parametrize(("utterance", "intent"), KEYWORD_CASES)
def test_every_keyword_maps(utterance: str, intent: str) -> None:
    assert match_intent(utterance) == intent


@pytest.mark.parametrize(
    ("utterance", "intent"),
    [
        ("Next!", "next"),
        ("  NEXT.  ", "next"),
        ("Go back", "prev"),
        ("СТОП.", "cancel"),
        ("Ещё раз?", "repeat"),
        ("repeat, please", "repeat"),
        ("ok, done", "cancel"),
    ],
)
def test_mixed_case_and_punctuation(utterance: str, intent: str) -> None:
    assert match_intent(utterance) == intent


@pytest.mark.parametrize(
    "utterance",
    [
        "what is the next step in this procedure",
        "can you repeat the second instruction please",
        "how do I go back to the previous menu on this device",
        "стоп а что если поменять щупы местами",
        "how do I check battery voltage",
    ],
)
def test_long_sentences_are_questions(utterance: str) -> None:
    assert match_intent(utterance) is None


@pytest.mark.parametrize("utterance", ["", "   ", "...", "hello", "multimeter", "раз"])
def test_non_commands_return_none(utterance: str) -> None:
    assert match_intent(utterance) is None
