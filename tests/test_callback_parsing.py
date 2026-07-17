"""Tests for :func:`feedback.parse_feedback_callback`.

The parser is pure and importable with no environment variables set (no bot
token, no database URL), so these tests need only pytest.
"""
import pytest

from feedback import FeedbackCallbackError, parse_feedback_callback


def test_parses_like():
    assert parse_feedback_callback("fb_like_1") == ("like", 1)


def test_parses_dislike():
    assert parse_feedback_callback("fb_dislike_42") == ("dislike", 42)


def test_missing_id_raises():
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("fb_like")


def test_non_numeric_id_raises():
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("fb_like_abc")


def test_unknown_action_raises():
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("fb_love_1")


def test_extra_underscores_raises():
    # split("_", 2) yields id="1_2"; an id with stray segments must be rejected.
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("fb_like_1_2")


def test_wrong_prefix_raises():
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("xx_like_1")


def test_empty_string_raises():
    with pytest.raises(FeedbackCallbackError):
        parse_feedback_callback("")


def test_error_is_a_value_error():
    # Typed error stays a ValueError subclass for lenient except handlers.
    with pytest.raises(ValueError):
        parse_feedback_callback("garbage")
