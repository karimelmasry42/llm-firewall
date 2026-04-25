"""
FastAPI application factory.

Creates the firewall app, wires shared state (settings, classifier specs,
preloaded validators, decision log) and registers the route modules.
"""
from __future__ import annotations

from fastapi import FastAPI

from llm_firewall.api import dashboard as dashboard_routes
from llm_firewall.api import routes as chat_routes
from llm_firewall.api._processing import (
    flatten_classifier_specs,
    preload_validators,
)
from llm_firewall.classifiers.registry import (
    ClassifierSpec,
    get_input_classifier_specs_by_language,
    get_output_classifier_specs,
)
from llm_firewall.core.config import Settings


def create_app(
    settings: Settings | None = None,
    input_classifier_specs: list[ClassifierSpec] | None = None,
    input_classifier_specs_by_language: dict[str, list[ClassifierSpec]] | None = None,
    output_classifier_specs: list[ClassifierSpec] | None = None,
) -> FastAPI:
    """Create the FastAPI app with lazy-loaded validators."""
    app = FastAPI(
        title="PromptShield",
        description="Routes prompts through input/output classifier ensembles.",
        version="0.2.0",
    )

    app.state.settings = settings or Settings()
    app.state.decision_log = []

    routed_input_specs = (
        {
            language: list(specs)
            for language, specs in input_classifier_specs_by_language.items()
        }
        if input_classifier_specs_by_language is not None
        else (
            {"en": list(input_classifier_specs)}
            if input_classifier_specs is not None
            else get_input_classifier_specs_by_language()
        )
    )
    app.state.input_validators = {}
    app.state.output_validator = None
    app.state.input_classifier_specs_by_language = routed_input_specs
    app.state.input_classifier_specs = flatten_classifier_specs(routed_input_specs)
    app.state.output_classifier_specs = list(
        output_classifier_specs or get_output_classifier_specs()
    )

    preload_validators(app)

    app.include_router(chat_routes.router)
    app.include_router(dashboard_routes.router)

    return app


app = create_app()


def run() -> None:
    """Console-script entry point for `llm-firewall`."""
    import uvicorn

    uvicorn.run(
        "llm_firewall.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
