from typing import Any

from backend.agents.agent import (
    get_reference_context,
    process_user_message,
    record_exchange,
)
from backend.services.image_generation import (
    analyze_canvas_page,
    generate_canvas_image,
    image_generation_enabled,
)
from backend.services.llm import unload_qwen_model

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
    typed_text: str = "",
    inference_steps: int = 10,
) -> dict[str, Any]:
    typed_text = typed_text.strip()
    conversation_history, relevant_memories = get_reference_context(note_id, typed_text)
    analysis = analyze_canvas_page(
        page_data_url,
        stroke_count=len(strokes),
        typed_text=typed_text,
        conversation_history=conversation_history,
        relevant_memories=relevant_memories,
    )
    action = str(analysis["action"])
    requires_deep_thinking = bool(analysis["requires_deep_thinking"])
    user_text = typed_text or str(analysis["display_text"])
    conversation_message = str(analysis["conversation_message"]).strip() or user_text
    history_message = typed_text if typed_text and not strokes else conversation_message
    text_response: str | None = None
    image_url: str | None = None

    if action == "clarify":
        text_response = str(analysis["clarification_question"])
        record_exchange(note_id, history_message, text_response)

    elif action == "chat":
        if requires_deep_thinking:
            text_response = "".join(
                process_user_message(note_id, history_message, think=True)
            ).strip()
        else:
            text_response = str(analysis["text_response"]).strip()
            record_exchange(
                note_id,
                history_message,
                text_response,
                remember=True,
            )

    elif action == "both":
        if requires_deep_thinking:
            text_response = "".join(
                process_user_message(
                    note_id,
                    history_message,
                    system_context=IMAGE_TOOL_CONTEXT,
                    record=False,
                    think=True,
                )
            ).strip()
        else:
            text_response = str(analysis["text_response"]).strip()

    if action in {"image", "both"}:
        if image_generation_enabled():
            unload_qwen_model()
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
        record_exchange(
            note_id,
            history_message,
            text_response,
            remember=True,
        )

    return {
        "intent": action,
        "confidence": analysis["confidence"],
        "deep_thinking": requires_deep_thinking,
        "user_text": user_text,
        "text_response": text_response,
        "image_url": image_url,
        "image_prompt": analysis["image_prompt"] if image_url else None,
        "inference_steps": inference_steps if image_url else None,
    }
