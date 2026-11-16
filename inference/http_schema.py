import re
from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

DATE_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
DATETIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:[.,]\d{1,6})?)?(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)?",
    flags=re.ASCII,
)


def require_iso_datetime(value: str) -> str:
    if DATETIME_PATTERN.fullmatch(value) is None:
        raise ValueError("Expected an ISO datetime with a time component")
    datetime.fromisoformat(value)
    return value


def require_fact_date(value: str) -> str:
    if DATE_PATTERN.fullmatch(value):
        date.fromisoformat(value)
        return value
    try:
        return require_iso_datetime(value)
    except ValueError as exc:
        raise ValueError("Expected a valid ISO date or datetime") from exc


def require_utf8(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Expected valid Unicode text") from exc
    return value


def require_polarity(value: object) -> int:
    if type(value) is not int or value not in (-1, 1):
        raise ValueError("Polarity must be -1 or 1")
    return value


def require_text(value: str) -> str:
    if not value.strip():
        raise ValueError("Expected a non-blank string")
    return value


NonBlank = Annotated[str, AfterValidator(require_text), AfterValidator(require_utf8)]
IsoDatetime = Annotated[str, AfterValidator(require_iso_datetime)]
FactDate = Annotated[str, AfterValidator(require_fact_date)]
Index = Annotated[int, Field(ge=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", allow_inf_nan=False)


class Turn(ContractModel):
    idx: Index
    speaker: NonBlank
    text: Annotated[str, AfterValidator(require_utf8)]


class Series(ContractModel):
    account_id: str
    call_index: int
    series_len: int


class ExtractRequest(BaseModel):
    # Full dataset rows also work. Only the declared fields are retained.
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    call_id: NonBlank
    started_at: IsoDatetime
    turns: Annotated[list[Turn], Field(min_length=1)]
    duration_sec: Annotated[float, Field(ge=0)] | None = None
    series: Series | None = None
    noise_level: float | None = None

    @model_validator(mode="after")
    def unique_turn_indices(self) -> Self:
        if len({turn.idx for turn in self.turns}) != len(self.turns):
            raise ValueError("Turn indices must be unique")
        return self


class FactObject(ContractModel):
    name: str = Field(description="Use an empty string when the object is unnamed")
    id: str | None = None
    type: str | None = None


class Source(ContractModel):
    turn_idx: Index
    span: Annotated[list[int], Field(min_length=2, max_length=2)] | None = None
    quote: str | None = None


class Fact(ContractModel):
    object: FactObject
    attribute: Literal["service_state", "action", "agreement", "tariff", "ticket_status"]
    value: NonBlank
    polarity: Annotated[Literal[-1, 1], BeforeValidator(require_polarity)]
    speaker: str | None = None
    valid_from: FactDate
    valid_to: FactDate | None
    source: Source
    confirmed: bool | None = None

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        # ISO dates are validated above. v2.1 allows transitions on the same calendar day.
        if self.valid_to is not None and self.valid_to[:10] < self.valid_from[:10]:
            raise ValueError("valid_to must not be on an earlier day than valid_from")
        return self


class ExtractResponse(ContractModel):
    model_config = ConfigDict(extra="forbid")
    call_id: NonBlank
    facts: list[Fact]
