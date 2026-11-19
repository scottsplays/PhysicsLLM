"""Check the three API keys in .env and list the models each one can use.

Run from the project folder:
    .venv\\Scripts\\python.exe check_keys.py

Uses only the Python standard library. Prints and saves results to
key_check.txt. Key values are never printed or written.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_env(path):
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        env[name.strip()] = value.strip().strip('"').strip("'")
    return env


def get_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    env = load_env(HERE / ".env")
    lines = []

    key = env.get("OPENAI_API_KEY", "")
    status, data = get_json("https://api.openai.com/v1/models",
                            {"Authorization": f"Bearer {key}"})
    if status == 200:
        ids = sorted(m["id"] for m in data["data"]
                     if m["id"].startswith(("gpt", "o")))
        lines.append(f"OPENAI  OK  ({len(ids)} chat models): " + ", ".join(ids))
    else:
        lines.append(f"OPENAI  FAILED  status={status}  {data}")

    key = env.get("GEMINI_API_KEY", "")
    status, data = get_json(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        {"x-goog-api-key": key})
    if status == 200:
        ids = [m["name"].replace("models/", "") for m in data.get("models", [])
               if "generateContent" in m.get("supportedGenerationMethods", [])]
        lines.append(f"GEMINI  OK  ({len(ids)} models): " + ", ".join(ids))
    else:
        lines.append(f"GEMINI  FAILED  status={status}  {data}")

    key = env.get("ANTHROPIC_API_KEY") or env.get("CLAUDE_API_KEY", "")
    status, data = get_json("https://api.anthropic.com/v1/models?limit=100",
                            {"x-api-key": key,
                             "anthropic-version": "2023-06-01"})
    if status == 200:
        ids = [m["id"] for m in data.get("data", [])]
        lines.append(f"CLAUDE  OK  ({len(ids)} models): " + ", ".join(ids))
    else:
        lines.append(f"CLAUDE  FAILED  status={status}  {data}")

    text = "\n\n".join(lines)
    print(text)
    (HERE / "key_check.txt").write_text(text + "\n", encoding="utf-8")
    print("\nSaved to key_check.txt")


if __name__ == "__main__":
    main()
