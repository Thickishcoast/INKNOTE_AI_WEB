from typing import Any


def normalize_canvas_analysis(
    data: dict[str, Any],
    confidence_threshold: float,
) -> dict[str, object]:
    action = str(data["action"]).strip().lower()
    confidence = float(data["confidence"])
    text_response = str(data["text_response"]).strip()
    image_prompt = str(data["image_prompt"]).strip()
    clarification_question = str(data["clarification_question"]).strip()
    requires_deep_thinking = bool(data["requires_deep_thinking"])

    if confidence < confidence_threshold:
        action = "clarify"
    if action in {"image", "both"} and not image_prompt:
        action = "clarify"
    if action == "chat" and not text_response:
        requires_deep_thinking = True
    if action == "clarify" and not clarification_question:
        clarification_question = (
            "Should I answer your canvas as a question, generate an image, or do both?"
        )

    return {
        "action": action,
        "confidence": confidence,
        "requires_deep_thinking": requires_deep_thinking,
        "recognized_text": [
            item.strip() for item in data["recognized_text"] if item.strip()
        ],
        "display_text": str(data["display_text"]).strip() or "Canvas submission",
        "conversation_message": str(data["conversation_message"]).strip(),
        "text_response": text_response,
        "image_prompt": image_prompt,
        "clarification_question": clarification_question,
    }
