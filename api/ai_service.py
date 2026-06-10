import logging

from config import get_settings

logger = logging.getLogger(__name__)

DISEASE_CONTEXT = {
    "Early Blight": (
        "Early blight (Alternaria solani) causes dark brown spots with concentric rings "
        "on older leaves, yellowing, and defoliation."
    ),
    "Late Blight": (
        "Late blight (Phytophthora infestans) causes water-soaked lesions, white mold "
        "on leaf undersides, and rapid plant collapse in humid conditions."
    ),
    "Healthy": (
        "The leaf appears healthy with no significant disease symptoms detected."
    ),
}

# Tried in order — gemini-2.0-flash often has 0 free-tier quota on new projects
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


def get_ai_advice(disease: str, confidence: float) -> dict:
    settings = get_settings()
    context = DISEASE_CONTEXT.get(disease, "Unknown potato leaf condition.")

    if not settings.gemini_api_key:
        return {
            "advice": _fallback_advice(disease, confidence, "GEMINI_API_KEY is not set in api/.env"),
            "source": "fallback",
        }

    prompt = f"""You are an agricultural expert helping potato farmers.

The ML model classified a potato leaf image as: {disease}
Confidence: {confidence * 100:.1f}%
Disease context: {context}

Provide a concise, practical response with these sections:
1. **Summary** - One sentence about the diagnosis
2. **Symptoms** - What to look for
3. **Treatment** - Immediate actions and fungicides/organic options
4. **Prevention** - How to avoid spread
5. **When to consult an expert**

Keep the response under 300 words. Use plain language for farmers."""

    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)

            if response.text and response.text.strip():
                return {"advice": response.text.strip(), "source": "gemini"}

            last_error = f"{model_name} returned empty response"
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
            logger.warning("Gemini model failed: %s", last_error)

    return {
        "advice": _fallback_advice(
            disease,
            confidence,
            f"Gemini API error — {last_error}",
        ),
        "source": "fallback",
    }


def _fallback_advice(disease: str, confidence: float, reason: str = "") -> str:
    tips = {
        "Early Blight": (
            "Remove infected leaves, improve airflow, avoid overhead watering, and apply "
            "copper-based or chlorothalonil fungicides. Rotate crops and use certified seed."
        ),
        "Late Blight": (
            "Act immediately: destroy severely infected plants, apply mancozeb or "
            "metalaxyl fungicides, and avoid wet foliage. This disease spreads very fast."
        ),
        "Healthy": (
            "Continue regular monitoring, maintain balanced fertilization, ensure good "
            "drainage, and scout weekly for early signs of blight."
        ),
    }
    tip = tips.get(disease, "Consult a local agricultural extension officer.")
    footer = f"_({reason})_" if reason else ""
    return (
        f"**Diagnosis:** {disease} ({confidence * 100:.1f}% confidence)\n\n"
        f"**Recommendation:** {tip}\n\n"
        f"{footer}"
    )
