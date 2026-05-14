"""Known Tata STRIVE center_name values in BigQuery intraining_students."""

from __future__ import annotations

TATA_STRIVE_CENTER_NAMES: tuple[str, ...] = (
    "Tata STRIVE - Aligarh",
    "Tata STRIVE - Hyderabad",
    "Tata STRIVE - Midnapore",
    "Tata STRIVE - Mumbai",
    "Tata STRIVE - Nashik",
    "Tata STRIVE - Pune",
)


def resolve_center_name(value: str) -> str | None:
    """Map free text or legacy underscore IDs to a catalog center_name."""
    raw = (value or "").strip()
    if not raw:
        return None
    if raw in TATA_STRIVE_CENTER_NAMES:
        return raw
    lowered = raw.lower()
    for name in TATA_STRIVE_CENTER_NAMES:
        if name.lower() == lowered:
            return name
    underscored = raw.replace("_", " ")
    for name in TATA_STRIVE_CENTER_NAMES:
        if name.lower() == underscored.lower():
            return name
    return None
