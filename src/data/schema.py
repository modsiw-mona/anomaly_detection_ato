"""Canonical login-event schema shared by every dataset.

Feature and model code only ever use the field names defined here. Each dataset maps its own
column names onto them in ``conf/dataset/<name>.yaml``, so supporting a new dataset means writing
a new mapping file, not changing code.

Minimum data requirements (see ``docs/datasets/README.md``): every dataset must provide all
``REQUIRED`` fields plus at least one ``CONTEXT`` field. Without ``user_id`` there is no per-user
behavioural baseline, which is the core of the approach.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    """How a canonical field participates in the minimum data requirements."""

    REQUIRED = "required"
    CONTEXT = "context"
    OPTIONAL = "optional"
    LABEL = "label"


class Kind(StrEnum):
    """Logical type of a field. Physical dtypes are decided when the data is loaded."""

    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    TEXT = "text"
    NUMERIC = "numeric"


@dataclass(frozen=True, slots=True)
class CanonicalField:
    """One field of the canonical login-event schema."""

    name: str
    role: Role
    kind: Kind
    description: str


FIELDS: tuple[CanonicalField, ...] = (
    CanonicalField("user_id", Role.REQUIRED, Kind.IDENTIFIER, "Account the login attempt targets."),
    CanonicalField("timestamp", Role.REQUIRED, Kind.TIMESTAMP, "When the login attempt happened."),
    CanonicalField("login_success", Role.REQUIRED, Kind.BOOLEAN, "Whether the login succeeded."),
    CanonicalField("ip", Role.CONTEXT, Kind.CATEGORICAL, "Client IP address."),
    CanonicalField(
        "asn",
        Role.CONTEXT,
        Kind.CATEGORICAL,
        "Autonomous system number of the IP (an identifier, not a quantity).",
    ),
    CanonicalField("device_type", Role.CONTEXT, Kind.CATEGORICAL, "Device class, e.g. mobile."),
    CanonicalField("user_agent", Role.CONTEXT, Kind.TEXT, "Raw user agent string."),
    CanonicalField("country", Role.OPTIONAL, Kind.CATEGORICAL, "Country derived from the IP."),
    CanonicalField("region", Role.OPTIONAL, Kind.CATEGORICAL, "Region derived from the IP."),
    CanonicalField("city", Role.OPTIONAL, Kind.CATEGORICAL, "City derived from the IP."),
    CanonicalField("os", Role.OPTIONAL, Kind.CATEGORICAL, "Operating system name and version."),
    CanonicalField("browser", Role.OPTIONAL, Kind.CATEGORICAL, "Browser name and version."),
    CanonicalField("rtt_ms", Role.OPTIONAL, Kind.NUMERIC, "Round-trip time in milliseconds."),
    CanonicalField(
        "label_ato", Role.LABEL, Kind.BOOLEAN, "Confirmed account takeover (incident response)."
    ),
    CanonicalField(
        "label_attack_ip", Role.LABEL, Kind.BOOLEAN, "IP found in a known-attacker list."
    ),
)

FIELDS_BY_NAME: dict[str, CanonicalField] = {field.name: field for field in FIELDS}
REQUIRED_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if f.role is Role.REQUIRED)
CONTEXT_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if f.role is Role.CONTEXT)
LABEL_FIELDS: frozenset[str] = frozenset(f.name for f in FIELDS if f.role is Role.LABEL)


class RequirementsError(ValueError):
    """A dataset does not meet the minimum data requirements."""


def check_requirements(fields: Iterable[str]) -> None:
    """Check that a set of canonical field names meets the minimum data requirements.

    Args:
        fields: Canonical field names a dataset provides (labels included or not).

    Raises:
        RequirementsError: If a name is unknown, a required field is missing, or no context
            field is present.
    """
    present = set(fields)
    unknown = present - FIELDS_BY_NAME.keys()
    if unknown:
        raise RequirementsError(f"Unknown canonical field(s): {sorted(unknown)}")
    missing = REQUIRED_FIELDS - present
    if missing:
        raise RequirementsError(
            f"Missing required field(s) {sorted(missing)}. Every dataset needs "
            f"{sorted(REQUIRED_FIELDS)}."
        )
    if not present & CONTEXT_FIELDS:
        raise RequirementsError(
            f"No context field present. Provide at least one of {sorted(CONTEXT_FIELDS)}."
        )
