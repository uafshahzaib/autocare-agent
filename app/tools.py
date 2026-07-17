"""LangChain tools available to the AutoCare agent."""
from __future__ import annotations

import re
from datetime import date

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.ingest import load_vector_store

# ---------------------------------------------------------------------------
# 1. RAG knowledge lookup tool
# ---------------------------------------------------------------------------


class KnowledgeQuery(BaseModel):
    query: str = Field(description="A natural-language question about vehicle maintenance.")


def _search_knowledge_base(query: str) -> str:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant maintenance documentation was found for this query."
    return "\n\n---\n\n".join(doc.page_content for doc in results)


def make_rag_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_search_knowledge_base,
        name="search_maintenance_docs",
        description=(
            "Search the internal vehicle maintenance knowledge base for facts about "
            "oil changes, brakes, tires, batteries, and dashboard warning lights. "
            "Use this whenever the user asks something that requires grounded, "
            "factual maintenance information rather than a calculation."
        ),
        args_schema=KnowledgeQuery,
    )


# ---------------------------------------------------------------------------
# 2. Deterministic service-interval calculator tool
# ---------------------------------------------------------------------------


class ServiceIntervalQuery(BaseModel):
    current_mileage_km: int = Field(description="The vehicle's current odometer reading in km.")
    last_service_mileage_km: int = Field(description="Odometer reading at the last full service, in km.")
    service_interval_km: int = Field(
        default=15000, description="Manufacturer-recommended service interval in km (default 15000)."
    )


def _service_interval_check(
    current_mileage_km: int, last_service_mileage_km: int, service_interval_km: int = 15000
) -> str:
    if current_mileage_km < last_service_mileage_km:
        return "Current mileage cannot be lower than the last service mileage — please check the inputs."

    driven_since_service = current_mileage_km - last_service_mileage_km
    remaining = service_interval_km - driven_since_service

    if remaining <= 0:
        return (
            f"Service is OVERDUE by {abs(remaining)} km. The vehicle has driven "
            f"{driven_since_service} km since the last service against a "
            f"{service_interval_km} km interval. Book a service as soon as possible."
        )
    if remaining <= 1000:
        return (
            f"Service is due soon: only {remaining} km remain until the next "
            f"{service_interval_km} km service interval."
        )
    return (
        f"No service due yet: {remaining} km remain until the next "
        f"{service_interval_km} km service interval."
    )


def make_service_interval_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_service_interval_check,
        name="check_service_interval",
        description=(
            "Deterministically calculate whether a vehicle service is due, based on "
            "current mileage, mileage at the last service, and the service interval. "
            "Use this for any question about whether/when a service is due."
        ),
        args_schema=ServiceIntervalQuery,
    )


# ---------------------------------------------------------------------------
# 3. Warning-light structured lookup tool
# ---------------------------------------------------------------------------

_WARNING_LIGHTS = {
    "engine": {
        "color": "amber",
        "meaning": "Onboard diagnostics detected an emissions or engine-management issue.",
        "urgency": "Address soon; if flashing, this indicates an active misfire — reduce load and get it inspected promptly to avoid catalytic converter damage.",
    },
    "oil": {
        "color": "red",
        "meaning": "Dangerously low oil pressure.",
        "urgency": "Stop the vehicle as soon as safely possible; continued driving risks severe engine damage.",
    },
    "battery": {
        "color": "red",
        "meaning": "The charging system is not maintaining voltage.",
        "urgency": "The vehicle may stall unexpectedly; have the charging system checked immediately.",
    },
    "brake": {
        "color": "red",
        "meaning": "Low brake fluid, engaged parking brake, or an ABS/brake system fault.",
        "urgency": "Check the parking brake and fluid level immediately before driving.",
    },
    "tire": {
        "color": "amber",
        "meaning": "One or more tires is significantly under-inflated.",
        "urgency": "Check tire pressures against the recommended values soon; not an immediate safety stop in most cases.",
    },
}


class WarningLightQuery(BaseModel):
    light_name: str = Field(
        description="Short keyword identifying the dashboard warning light, e.g. 'engine', 'oil', 'battery', 'brake', 'tire'."
    )


def _lookup_warning_light(light_name: str) -> str:
    key = re.sub(r"[^a-z]", "", light_name.lower())
    for name, info in _WARNING_LIGHTS.items():
        if name in key or key in name:
            return (
                f"{name.title()} warning light ({info['color']}): {info['meaning']} "
                f"Urgency: {info['urgency']}"
            )
    known = ", ".join(_WARNING_LIGHTS.keys())
    return f"Unknown warning light '{light_name}'. Known lights: {known}."


def make_warning_light_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_lookup_warning_light,
        name="lookup_warning_light",
        description=(
            "Look up the meaning and urgency of a dashboard warning light by name "
            "(e.g. 'engine', 'oil', 'battery', 'brake', 'tire')."
        ),
        args_schema=WarningLightQuery,
    )


# ---------------------------------------------------------------------------
# 4. Computer-vision tool: classify a dashboard warning-light photo
# ---------------------------------------------------------------------------


class ImageClassifyQuery(BaseModel):
    image_path: str = Field(description="Local file path to a photo of the dashboard warning light.")


def _classify_warning_light_image(image_path: str) -> str:
    from app.vision.classify import classify_image  # imported lazily: torch-heavy

    label, confidence = classify_image(image_path)
    return (
        f"Image classified as '{label}' warning light with {confidence:.0%} confidence. "
        f"Use lookup_warning_light('{label}') to get its meaning and urgency."
    )


def make_vision_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_classify_warning_light_image,
        name="classify_warning_light_image",
        description=(
            "Identify which dashboard warning light is shown in a photo, when the user "
            "provides an image instead of typing the light's name. Returns a class label "
            "(engine, oil, battery, brake, or tire) which can then be passed to "
            "lookup_warning_light for the full explanation."
        ),
        args_schema=ImageClassifyQuery,
    )


# ---------------------------------------------------------------------------
# 5. Risk-scoring / predictive-triage tool
#    Composes the service-interval and warning-light signals into a single
#    prioritized recommendation — this is the "impact" layer: it turns raw
#    facts into a decision (what should happen next, and how urgently).
# ---------------------------------------------------------------------------


class RiskAssessmentQuery(BaseModel):
    current_mileage_km: int = Field(description="The vehicle's current odometer reading in km.")
    last_service_mileage_km: int = Field(description="Odometer reading at the last full service, in km.")
    service_interval_km: int = Field(default=15000, description="Manufacturer-recommended service interval in km.")
    active_warning_light: str = Field(
        default="none",
        description="Name of an active dashboard warning light ('engine', 'oil', 'battery', 'brake', 'tire'), or 'none' if none is on.",
    )


def _assess_risk(
    current_mileage_km: int,
    last_service_mileage_km: int,
    service_interval_km: int = 15000,
    active_warning_light: str = "none",
) -> str:
    service_status = _service_interval_check(current_mileage_km, last_service_mileage_km, service_interval_km)
    is_overdue = "OVERDUE" in service_status
    is_due_soon = "due soon" in service_status

    light_color = None
    light_note = ""
    if active_warning_light and active_warning_light.lower() != "none":
        light_note = _lookup_warning_light(active_warning_light)
        key = re.sub(r"[^a-z]", "", active_warning_light.lower())
        for name, info in _WARNING_LIGHTS.items():
            if name in key or key in name:
                light_color = info["color"]
                break

    # Priority logic: a red light is always at least HIGH; combined with an
    # overdue service it becomes CRITICAL. Amber alone, or an overdue service
    # alone, is MEDIUM/HIGH. Otherwise LOW.
    if light_color == "red" and is_overdue:
        priority = "CRITICAL"
        action = "Stop using the vehicle for non-essential trips and book an urgent service today."
    elif light_color == "red":
        priority = "HIGH"
        action = "Have the vehicle inspected within the next 24-48 hours."
    elif is_overdue:
        priority = "HIGH"
        action = "Book a service appointment this week; the vehicle has exceeded its recommended interval."
    elif light_color == "amber" or is_due_soon:
        priority = "MEDIUM"
        action = "Schedule a service in the next 1-2 weeks; no immediate safety concern."
    else:
        priority = "LOW"
        action = "No action needed right now; continue routine monitoring."

    parts = [f"Priority: {priority}. Recommended action: {action}", f"Service check: {service_status}"]
    if light_note:
        parts.append(f"Warning light: {light_note}")
    return "\n".join(parts)


def make_risk_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_assess_risk,
        name="assess_maintenance_risk",
        description=(
            "Combine mileage/service status and any active warning light into a single "
            "prioritized (LOW/MEDIUM/HIGH/CRITICAL) recommendation for what the driver "
            "should do next. Use this when the user wants an overall risk assessment or "
            "'what should I do' answer, rather than a single isolated fact."
        ),
        args_schema=RiskAssessmentQuery,
    )


# ---------------------------------------------------------------------------
# Tool groupings used by the multi-agent architecture (see app/agents/)
# ---------------------------------------------------------------------------


def get_diagnostic_tools() -> list[StructuredTool]:
    """Tools for identifying/explaining a problem: RAG, vision, warning-light lookup."""
    return [make_rag_tool(), make_vision_tool(), make_warning_light_tool()]


def get_scheduling_tools() -> list[StructuredTool]:
    """Tools for deciding what to do about it: service math + risk triage."""
    return [make_service_interval_tool(), make_risk_tool()]


def get_all_tools() -> list[StructuredTool]:
    """Flat tool list — used by the single-agent version in app/agent.py."""
    return get_diagnostic_tools() + get_scheduling_tools()


def today() -> date:
    """Small helper kept separate so it can be mocked easily in tests."""
    return date.today()
