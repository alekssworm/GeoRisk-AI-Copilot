from app.schemas import (
    AdvancedRadiationFeatures,
    AdvancedScenarioInput,
    RAGAnswerResponse,
    RadiationFeatures,
    ScenarioInput,
)
from ml.explain import explain_prediction
from ml.classic.predict import compare_advanced_scenarios, predict_advanced_dose
from ml.predict import compare_scenarios, predict_dose
from rag.qa import RAGAssistant


def prediction_for(features: RadiationFeatures) -> dict:
    return predict_dose(features.to_feature_dict())


def compare_for(baseline: RadiationFeatures, scenarios: list[ScenarioInput]) -> list[dict]:
    scenario_payload = [
        {"name": scenario.name, "overrides": scenario.overrides} for scenario in scenarios
    ]
    return compare_scenarios(baseline.to_feature_dict(), scenario_payload)


def advanced_prediction_for(features: AdvancedRadiationFeatures) -> dict:
    return predict_advanced_dose(features.to_feature_dict())


def advanced_compare_for(
    baseline: AdvancedRadiationFeatures,
    scenarios: list[AdvancedScenarioInput],
) -> list[dict]:
    scenario_payload = [
        {"name": scenario.name, "overrides": scenario.overrides} for scenario in scenarios
    ]
    return compare_advanced_scenarios(baseline.to_feature_dict(), scenario_payload)


def explanation_for(features: RadiationFeatures) -> dict:
    return explain_prediction(features.to_feature_dict())


def rag_answer_for(question: str, top_k: int = 4) -> RAGAnswerResponse:
    answer = RAGAssistant().answer(question, top_k=top_k)
    return RAGAnswerResponse(**answer)


def generate_report(
    baseline: RadiationFeatures,
    scenarios: list[ScenarioInput],
    rag_question: str | None = None,
) -> dict:
    prediction = prediction_for(baseline)
    scenario_comparison = compare_for(baseline, scenarios)
    explanation = explanation_for(baseline)
    rag_answer = rag_answer_for(rag_question) if rag_question else None

    top_drivers = explanation.get("top_features", [])[:5]
    driver_lines = "\n".join(
        f"- {item['feature']}: {item['direction']} contribution ({item['value']:.3f})"
        for item in top_drivers
    )
    if not driver_lines:
        driver_lines = "- No feature explanation available."

    scenario_lines = "\n".join(
        f"- {item['name']}: {item['dose_rate_usv_h']:.3f} uSv/h "
        f"({item['risk_level']}, delta {item['delta_vs_baseline_usv_h']:+.3f})"
        for item in scenario_comparison
    )
    if not scenario_lines:
        scenario_lines = "- No alternative scenarios were supplied."

    rag_section = "No document question was included."
    if rag_answer is not None:
        citations = ", ".join(citation["citation_id"] for citation in rag_answer.citations)
        rag_section = f"{rag_answer.answer}\n\nCitations: {citations or 'none'}"

    report_markdown = f"""# GeoRisk AI Copilot Risk Analysis

## Executive Summary
The baseline predicted radiation dose rate is **{prediction["dose_rate_usv_h"]:.3f} uSv/h**.
The current risk level is **{prediction["risk_level"]}**. {prediction["advisory"]}

## Scenario Comparison
{scenario_lines}

## Main Risk Drivers
{driver_lines}

## Document-Grounded Context
{rag_section}

## Recommended Next Actions
- Validate contamination inputs against field survey measurements.
- Re-run the comparison with conservative rainfall and runoff assumptions.
- Use uploaded technical guidance to confirm intervention thresholds and monitoring cadence.
"""

    return {
        "report_markdown": report_markdown,
        "prediction": prediction,
        "scenario_comparison": scenario_comparison,
        "explanation": explanation,
        "rag_answer": rag_answer,
    }
