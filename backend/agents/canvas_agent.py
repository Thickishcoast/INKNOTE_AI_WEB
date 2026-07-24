from __future__ import annotations

from typing import Any

from backend.agents.agent import process_user_message, record_exchange
from backend.services.image_generation import (
    analyze_canvas_page,
    generate_canvas_image,
    image_generation_enabled,
)

IMAGE_TOOL_CONTEXT = """
InkNote has a FLUX image-generation tool and this request is being routed to it.
The image will be generated successfully before your text is shown to the user.
Answer only any separate written part of the request, concisely. Do not claim that
you cannot create images, do not offer manual drawing instructions, and do not
describe the requested scene merely as a substitute for generating it. Do not add
a generic image-completion acknowledgement.
""".strip()

def process_canvas_submission(
    note_id: str,
    page_data_url: str,
    strokes: list[dict[str, Any]],
    *,
    inference_steps: int = 10,
) -> dict[str, Any]:
    analysis = analyze_canvas_page(page_data_url, stroke_count=len(strokes))
    action = str(analysis["action"])
    user_text = str(analysis["display_text"])
    conversation_message = str(analysis["conversation_message"]).strip() or user_text
    text_response: str | None = None
    image_url: str | None = None

    if action == "clarify":
        text_response = str(analysis["clarification_question"])
        record_exchange(note_id, conversation_message, text_response)

    if action == "chat":
        text_response = "".join(process_user_message(note_id, conversation_message)).strip()

    if action == "both":
        text_response = "".join(
            process_user_message(
                note_id,
                conversation_message,
                system_context=IMAGE_TOOL_CONTEXT,
                record=False,
            )
        ).strip()

    if action in {"image", "both"}:
        if image_generation_enabled():
            recognized_text = list(analysis["recognized_text"])
            image_url = generate_canvas_image(
                str(analysis["image_prompt"]),
                recognized_text=recognized_text,
                num_inference_steps=inference_steps,
            )
        else:
            unavailable = "I understood this as an image request, but FLUX is currently disabled."
            text_response = f"{text_response}\n\n{unavailable}".strip() if text_response else unavailable
            action = "clarify"

    if action in {"image", "both"} and image_url and text_response:
        record_exchange(note_id, conversation_message, text_response)

    return {
        "intent": action,
        "confidence": analysis["confidence"],
        "user_text": user_text,
        "text_response": text_response,
        "image_url": image_url,
        "image_prompt": analysis["image_prompt"] if image_url else None,
        "inference_steps": inference_steps if image_url else None,
    }
