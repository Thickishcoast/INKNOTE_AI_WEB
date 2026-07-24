# Unified Canvas Agent Flow

## 1. Browser

The canvas stores every pen and eraser stroke. Undo moves the latest stroke to a redo
stack. Redo moves it back. Starting a new stroke clears the redo stack.

When Send is pressed, the browser sends:

- a white-background PNG of the complete canvas
- structured stroke data
- the current note ID

## 2. Qwen router

`analyze_canvas_page()` asks Qwen2.5-VL for strict JSON. The router distinguishes between
questions about a drawing and requests to generate a new image. A drawing alone does not
activate FLUX.

## 3. Tool execution

`process_canvas_submission()` performs the selected action:

- chat → conversation agent
- image → FLUX
- both → conversation agent, then FLUX
- clarify → a short clarification question

## 4. GPU lifecycle

Ollama receives `keep_alive=0`. FLUX loads only for image actions and is released after
generation when `INKNOTE_IMAGE_RELEASE_AFTER_GENERATION=true`.

## 5. AI canvas

The right canvas renders the latest turn. It can contain handwritten-style text, a generated
image, or both. The conversation dock retains the full saved history.
