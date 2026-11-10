"""Tests for the canonical schema and the minimum data requirements."""

import pytest

from data.schema import (
    CONTEXT_FIELDS,
    FIELDS,
    LABEL_FIELDS,
    REQUIRED_FIELDS,
    RequirementsError,
    check_requirements,
)

MINIMAL = {"user_id", "timestamp", "login_success", "asn"}


def test_field_names_are_unique() -> None:
    """Every field of the canonical schema has a distinct name."""
    names = [field.name for field in FIELDS]
    assert len(names) == len(set(names))


def test_role_groups_are_disjoint() -> None:
    """No field is in more than one of the required, context and label groups."""
    assert not REQUIRED_FIELDS & CONTEXT_FIELDS
    assert not REQUIRED_FIELDS & LABEL_FIELDS
    assert not CONTEXT_FIELDS & LABEL_FIELDS


def test_requirements_match_the_documented_minimum() -> None:
    """Required and context fields match the documented minimum."""
    assert {"user_id", "timestamp", "login_success"} == REQUIRED_FIELDS
    assert {"ip", "asn", "device_type", "user_agent"} == CONTEXT_FIELDS


@pytest.mark.parametrize("context", sorted(CONTEXT_FIELDS))
def test_any_single_context_field_is_enough(context: str) -> None:
    """The required fields plus any one context field meet the requirements."""
    check_requirements(REQUIRED_FIELDS | {context})


def test_minimal_dataset_passes() -> None:
    """The minimal set of fields meets the requirements."""
    check_requirements(MINIMAL)


@pytest.mark.parametrize("required", sorted(REQUIRED_FIELDS))
def test_missing_required_field_fails(required: str) -> None:
    """Dropping any required field raises ``RequirementsError`` naming it."""
    with pytest.raises(RequirementsError, match=required):
        check_requirements(MINIMAL - {required})


def test_no_context_field_fails() -> None:
    """A dataset without a context field raises ``RequirementsError``."""
    with pytest.raises(RequirementsError, match="context"):
        check_requirements(REQUIRED_FIELDS | {"country", "rtt_ms"})


def test_unknown_field_fails() -> None:
    """A field outside the canonical schema raises ``RequirementsError``."""
    with pytest.raises(RequirementsError, match="Unknown"):
        check_requirements(MINIMAL | {"favourite_colour"})
