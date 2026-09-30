import os
import json
import re
import time
import requests
from llama_cpp import Llama

INPUT_PATH = "/input/tasks.json"
OUTPUT_PATH = "/output/results.json"
MODEL_PATH = "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

TOTAL_BUDGET_SECONDS = 540
API_TIMEOUT_SECONDS = 16

SYSTEM_PROMPTS = {
    "FACTUAL": (
        "Answer accurately and directly. Follow any format, length, or style "
        "constraint in the user prompt. Do not add a conversational preamble."
    ),
    "MATH": (
        "Solve the problem accurately. Return a concise final answer. Show only "
        "minimal working when it is necessary to make the answer clear."
    ),
    "SENTIMENT": (
        "Classify the sentiment of the supplied text. Use the labels requested "
        "by the user. If no labels are specified, answer only: Positive, "
        "Negative, Neutral, or Mixed."
    ),
    "SUMMARY": (
        "Summarize only the supplied text. Strictly follow any requested "
        "sentence count, word limit, format, or length constraint. "
        "Output only the summary."
    ),
    "NER": (
        "Extract the requested named entities. Output only valid JSON with no "
        "Markdown. Follow the user's requested schema exactly. If no schema is "
        "given, return a JSON array of objects with text and type fields."
    ),
    "DEBUG": (
        "Fix the supplied code. Return only the complete corrected code. "
        "Do not use Markdown fences and do not include explanations."
    ),
    "LOGIC": (
        "Solve the constraint problem carefully. Return only a concise final "
        "answer unless the user explicitly requests reasoning."
    ),
    "CODE_GEN": (
        "Write a correct, complete implementation meeting the specification. "
        "Return only raw code, with no Markdown fences or explanation."
    ),
}

API_TASKS = {"NER", "DEBUG", "LOGIC", "CODE_GEN"}

API_MAX_TOKENS = {
    "NER": 140,
    "DEBUG": 220,
    "LOGIC": 110,
    "CODE_GEN": 240,
}

LOCAL_MAX_TOKENS = {
    "FACTUAL": 90,
    "MATH": 110,
    "SENTIMENT": 45,
    "SUMMARY": 120,
}

print("Loading local Llama-3.2 model...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1536,
    n_threads=2,
    n_batch=256,
    verbose=False,
)
print("Model loaded.")


def analyze_intent(prompt):
    p = prompt.lower()

    if any(x in p for x in (
        "summarize", "summarise", "summary", "condense",
        "tl;dr", "brief summary", "shorten"
    )):
        return "SUMMARY"

    if any(x in p for x in (
        "sentiment", "classify the tone", "classify the emotion",
        "positive or negative", "opinion expressed"
    )):
        return "SENTIMENT"

    code_markers = (
        "def ", "function", "code", "script", "snippet",
        "class ", "sql", "bash", "program"
    )
    bug_markers = (
        "bug", "fix", "error", "wrong", "fails",
        "failed", "incorrect", "debug", "issue"
    )

    if any(x in p for x in code_markers) and any(x in p for x in bug_markers):
        return "DEBUG"

    if any(x in p for x in (
        "write a", "create a", "implement", "generate",
        "develop", "build a"
    )) and any(x in p for x in code_markers):
        return "CODE_GEN"

    if any(x in p for x in (
        "named entity", "named entities", "extract entities",
        "extract all entities", "entities and their types"
    )):
        return "NER"

    if any(x in p for x in (
        "logic puzzle", "all conditions", "each owns",
        "who owns", "truth", "lies", "deduce",
        "constraint", "different pet"
    )):
        return "LOGIC"

    if any(x in p for x in (
        "calculate", "how many", "percentage", "probability",
        "equation", "derivative", "evaluate", "solve for",
        "remain", "total", "value of"
    )):
        return "MATH"

    return "FACTUAL"


def sanitize_output(raw_text, task_type):
    cleaned = (raw_text or "").strip()

    cleaned = re.sub(
        r"^```(?:json|python|javascript|typescript|sql|bash|text)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    if task_type == "NER":
        try:
            parsed = json.loads(cleaned)
            return json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return "[]"

    return cleaned


def call_fireworks_api(system_prompt, user_prompt, task_type):
    api_key = os.environ.get("FIREWORKS_API_KEY")
    base_url = os.environ.get("FIREWORKS_BASE_URL", "").rstrip("/")
    allowed_models = os.environ.get("ALLOWED_MODELS", "")

    if not api_key or not base_url or not allowed_models:
        return ""

    target_model = allowed_models.split(",")[0].strip()

    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": API_MAX_TOKENS[task_type],
    }

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
        print(f"API failure for {task_type}: {exc}")
        return ""


def call_local(system_prompt, user_prompt, task_type):
    formatted_prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    output = llm(
        formatted_prompt,
        max_tokens=LOCAL_MAX_TOKENS.get(task_type, 100),
        temperature=0.0,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    return output["choices"][0]["text"].strip()


def route_and_solve(task_prompt, started_at):
    task_type = analyze_intent(task_prompt)
    system_prompt = SYSTEM_PROMPTS[task_type]

    elapsed = time.monotonic() - started_at
    remaining = TOTAL_BUDGET_SECONDS - elapsed

    if task_type in API_TASKS and remaining > API_TIMEOUT_SECONDS + 12:
        print(f" -> API: {task_type}")
        answer = call_fireworks_api(system_prompt, task_prompt, task_type)
        if answer:
            return sanitize_output(answer, task_type)

        print(" -> API unavailable; using local fallback")

    print(f" -> LOCAL: {task_type}")
    try:
        answer = call_local(system_prompt, task_prompt, task_type)
        return sanitize_output(answer, task_type)
    except Exception as exc:
        print(f" -> Local inference failure: {exc}")
        return "[]" if task_type == "NER" else ""


def write_results(results):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    started_at = time.monotonic()

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"{INPUT_PATH} not found")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    print(f"Processing {len(tasks)} tasks...")

    for index, task in enumerate(tasks, start=1):
        task_id = task.get("task_id", "")
        prompt = task.get("prompt", "")

        if time.monotonic() - started_at >= TOTAL_BUDGET_SECONDS:
            print("Time budget reached; writing completed results.")
            break

        print(f"[{index}/{len(tasks)}] {task_id}")

        try:
            answer = route_and_solve(prompt, started_at)
        except Exception as exc:
            print(f"Task failure: {exc}")
            answer = ""

        results.append({
            "task_id": task_id,
            "answer": answer,
        })


        write_results(results)

    write_results(results)
    print(f"Done. Wrote {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()