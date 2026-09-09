#!/usr/bin/env python3
"""
Galileo - An Ollama-backed agentic chatbot with persistent memory.

Features:
- Chained tool calls with multiple thinking steps.
- Persistent memory stored in memory.md (key-value).
- Configurable via config.json.
- Editable system prompt in .prompt.md.
- API-key-free tools: weather, news, search, etc.

Run:
    python3 galileo.py

Requires:
    pip install ollama ddgs trafilatura requests feedparser
    ollama pull qwen3:0.6b   (or change MODEL in config)
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ollama import chat

# ───────────────────────────── Config loading ─────────────────────────────

DEFAULT_CONFIG = {
    "model": "qwen3:0.6b",
    "max_tool_hops": 20,
    "max_identical_repeats": 3,
    "tool_timeout_seconds": 30,
    "max_output_chars": 6000,
    "max_fetch_chars": 8000,
    "console_preview_chars": 300,
    "memory_file": "memory.md",
    "prompt_file": ".prompt.md",
    "enabled_tools": {
        "get_current_time": True,
        "calculator": True,
        "web_search": True,
        "get_website_content": True,
        "run_terminal_command": True,
        "get_seconds_until_datetime": True,
        "extract_number_from_text": True,
        "get_weather": True,
        "get_news": True,
        "get_memory": True,
        "set_memory": True,
        "list_memory": True,
    }
}

CONFIG_FILE = "config.json"

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        return DEFAULT_CONFIG.copy()

CONFIG = load_config()

# ───────────────────────────── Prompt loading ─────────────────────────────

DEFAULT_PROMPT = """You are Galileo, a helpful, curious, and slightly witty AI assistant.

You operate as an agent: you may call a tool, look at its result, then decide to call another tool based on what you learned, and keep doing that for as many steps as the task actually requires before giving your final answer. Do not stop after one tool call just because you got a result — check whether it actually answers the user's question, and if you need more information or another step (e.g. searching, then fetching a page from the results, then extracting a number from it, then calculating with it), do it.

You have access to a persistent memory stored in a markdown file. You can remember facts, user preferences, or past answers across sessions. Use the tools:
- get_memory(key) – retrieve a value.
- set_memory(key, value) – store or update a value.
- list_memory() – see all stored keys.

Available tools:
- get_current_time (no inputs)
- calculator(expression)
- web_search(query, max_results)
- get_website_content(url)
- run_terminal_command(command, timeout)
- get_seconds_until_datetime(datetime_str) - preferred for time-until calculations
- extract_number_from_text(text) - pulls the first number out of a string
- get_weather(location) - current weather conditions
- get_news(query) - latest headlines from Google News (optional search query)
- get_memory(key)
- set_memory(key, value)
- list_memory()

Typical multi-step patterns:
  * web_search -> get_website_content on the best result -> extract_number_from_text on the relevant sentence -> calculator
  * get_current_time -> get_seconds_until_datetime for a deadline
  * get_weather -> get_news (if weather affects news)
  * get_memory to recall past information, then use that in a calculator or search.

You may think between tool calls as many times as needed. Only stop calling tools once you actually have what you need to answer. Keep your final answer clear, concise, and engaging."""

PROMPT_FILE = CONFIG.get("prompt_file", ".prompt.md")

def load_prompt() -> str:
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r") as f:
            return f.read()
    else:
        with open(PROMPT_FILE, "w") as f:
            f.write(DEFAULT_PROMPT)
        return DEFAULT_PROMPT

SYSTEM_PROMPT = load_prompt()

# ───────────────────────────── Styling ─────────────────────────────

BOLD_GRAY = "\033[1;90m"
GRAY = "\033[90m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"

# ───────────────────────────── Memory tools ─────────────────────────────

def _read_memory() -> Dict[str, str]:
    """Read key-value pairs from memory.md."""
    memory_file = CONFIG.get("memory_file", "memory.md")
    data = {}
    if os.path.exists(memory_file):
        with open(memory_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("- "):
                    # Format: "- key: value"
                    parts = line[2:].split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        data[key] = value
    return data

def _write_memory(data: Dict[str, str]) -> None:
    """Write key-value pairs to memory.md."""
    memory_file = CONFIG.get("memory_file", "memory.md")
    with open(memory_file, "w") as f:
        for key, value in data.items():
            f.write(f"- {key}: {value}\n")

def get_memory(key: str) -> str:
    """Retrieve a value from persistent memory."""
    data = _read_memory()
    return data.get(key, "Key not found")

def set_memory(key: str, value: str) -> str:
    """Store a key-value pair in persistent memory (overwrites if exists)."""
    data = _read_memory()
    data[key] = value
    _write_memory(data)
    return f"Stored '{key}' = '{value}'"

def list_memory() -> str:
    """List all keys stored in memory."""
    data = _read_memory()
    if not data:
        return "Memory is empty."
    return ", ".join(data.keys())

# ───────────────────────────── Core Tools ─────────────────────────────

def get_current_time() -> str:
    """Return current date and time as a formatted string."""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M:%S %p")

def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    safe_dict = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "exp": math.exp, "pi": math.pi, "e": math.e, "pow": pow,
    }
    allowed_chars = re.compile(r"^[0-9+\-*/%().\s,a-zA-Z_]+$")
    if not expression or not allowed_chars.match(expression):
        return "Error: expression contains invalid characters"
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return a list of results with title, link, and snippet."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: 'ddgs' package not installed. Install with: pip install ddgs"
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = 5
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or "No title"
            href = r.get("href") or r.get("link") or ""
            body = r.get("body") or r.get("snippet") or ""
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"

def get_website_content(url: str) -> str:
    """Fetch and extract main content from a URL as Markdown."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"
    try:
        import trafilatura
    except ImportError:
        return "Error: 'trafilatura' package not installed. Install with: pip install trafilatura"
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: could not download content from {url}"
        markdown = trafilatura.extract(
            downloaded,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not markdown:
            return "Error: no main content could be extracted."
        max_fetch = CONFIG.get("max_fetch_chars", 8000)
        if len(markdown) > max_fetch:
            markdown = markdown[:max_fetch] + "\n\n...[content truncated]..."
        return markdown
    except Exception as e:
        return f"Error fetching page: {e}"

# ─── Hardened terminal command safety ──────────────────────────────

_SAFE_COMMAND_NAMES = {
    "ls", "pwd", "whoami", "date", "cal", "echo", "printf",
    "cat", "head", "tail", "wc", "grep", "find", "file", "stat",
    "df", "du", "free", "uptime", "uname", "hostname",
    "curl", "wget",
    "python3", "python", "node", "npm", "pip", "pip3",
    "git",
    "which", "env", "history",
    "sort", "uniq", "cut", "awk", "sed", "diff", "tr",
}

_DISALLOWED_SHELL_SYNTAX = re.compile(r"(;|&&|\|\||`|\$\(|>>|>|<|\||&(?!&))")

_DANGEROUS_FLAG_PATTERNS = [
    re.compile(r"\brm\b"),
    re.compile(r"\bdd\b"),
    re.compile(r"\bmkfs"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bsu\b"),
    re.compile(r"\bchmod\b.*\b777\b"),
    re.compile(r"\bchown\b"),
    re.compile(r":\(\)\s*\{"),          # fork bomb pattern :(){ :|:& };:
    re.compile(r"/dev/(sd|nvme|hd)"),   # raw block devices
    re.compile(r"--exec\b"),
    re.compile(r"-o\s+\S"),             # curl/wget writing to a file
    re.compile(r"--output\b"),
    re.compile(r"-O\b"),
]

def _is_command_safe(command: str) -> tuple[bool, str]:
    if not command or not command.strip():
        return False, "empty command"

    if _DISALLOWED_SHELL_SYNTAX.search(command):
        return False, (
            "command chaining/redirection/subshells are not allowed "
            "(;, &&, ||, |, `, $(), >, >>, <)"
        )

    for pattern in _DANGEROUS_FLAG_PATTERNS:
        if pattern.search(command):
            return False, f"blocked dangerous pattern: {pattern.pattern}"

    try:
        tokens = shlex.split(command)
    except ValueError as e:
        return False, f"could not parse command: {e}"

    if not tokens:
        return False, "empty command"

    base_cmd = tokens[0].split("/")[-1]
    if base_cmd not in _SAFE_COMMAND_NAMES:
        return False, (
            f"'{base_cmd}' is not in the allowlist of safe commands "
            f"({', '.join(sorted(_SAFE_COMMAND_NAMES))})"
        )

    return True, ""

def run_terminal_command(command: str, timeout: int = None) -> str:
    """Execute a shell command from a safety allowlist and return its output."""
    if timeout is None:
        timeout = CONFIG.get("tool_timeout_seconds", 30)
    ok, reason = _is_command_safe(command)
    if not ok:
        return f"Error: command blocked ({reason})"

    try:
        timeout = max(1, min(int(timeout), 120))
    except (TypeError, ValueError):
        timeout = CONFIG.get("tool_timeout_seconds", 30)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return "(no output)"
        max_out = CONFIG.get("max_output_chars", 6000)
        if len(output) > max_out:
            output = output[:max_out] + "\n\n...[output truncated]..."
        return output.strip()
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"

def get_seconds_until_datetime(datetime_str: str) -> str:
    """
    Calculate the number of seconds from now until the given datetime.
    Input format: "YYYY-MM-DD HH:MM:SS" (e.g., "2026-03-13 00:00:00").
    """
    try:
        target = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Error: datetime must be in format 'YYYY-MM-DD HH:MM:SS'"
    now = datetime.now()
    seconds = int((target - now).total_seconds())
    if seconds < 0:
        return "Error: target datetime is in the past"
    return str(seconds)

def extract_number_from_text(text: str) -> str:
    """Extract the first floating-point (or integer) number from the given text."""
    if not isinstance(text, str):
        return "0"
    match = re.search(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\.\d+", text)
    if match:
        return match.group().replace(",", "")
    return "0"

def get_weather(location: str) -> str:
    """Fetch current weather for a location using wttr.in."""
    try:
        import requests
    except ImportError:
        return "Error: 'requests' package not installed. Install with: pip install requests"

    if not location or not location.strip():
        return "Error: location cannot be empty"

    url = f"https://wttr.in/{location.strip()}?format=%l:+%c,+%t,+%w,+%h"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        weather = resp.text.strip()
        if not weather or "Sorry" in weather:
            return f"No weather data found for '{location}'"
        return weather
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather: {e}"

def get_news(query: str = "") -> str:
    """Fetch top headlines or search results from Google News RSS."""
    try:
        import feedparser
    except ImportError:
        return "Error: 'feedparser' package not installed. Install with: pip install feedparser"

    if query and query.strip():
        q = query.strip().replace(" ", "%20")
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    else:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
    except Exception as e:
        return f"Error fetching news: {e}"

    if not feed.entries:
        return "No news found."

    entries = feed.entries[:7]
    lines = []
    for entry in entries:
        title = entry.title.strip()
        link = entry.link if entry.link else entry.guid
        lines.append(f"• {title}\n  {link}")
    return "\n\n".join(lines)

# ──────────────────────── Tool registration ──────────────────────

# All tool functions (including memory tools)
ALL_TOOLS = [
    get_current_time,
    calculator,
    web_search,
    get_website_content,
    run_terminal_command,
    get_seconds_until_datetime,
    extract_number_from_text,
    get_weather,
    get_news,
    get_memory,
    set_memory,
    list_memory,
]

# Filter based on config
enabled_tools = CONFIG.get("enabled_tools", {})
TOOLS = [t for t in ALL_TOOLS if enabled_tools.get(t.__name__, True)]

AVAILABLE_FUNCTIONS: Dict[str, Callable[..., str]] = {f.__name__: f for f in TOOLS}

# ───────────────────────────── Agent loop ─────────────────────────────

@dataclass
class TurnStats:
    """Tracks what happened during one user turn, for transparency + loop detection."""
    hops: int = 0
    tool_calls_made: List[tuple[str, str]] = field(default_factory=list)

    def record(self, name: str, args_repr: str) -> None:
        self.hops += 1
        self.tool_calls_made.append((name, args_repr))

    def identical_repeat_count(self, name: str, args_repr: str) -> int:
        count = 0
        for n, a in reversed(self.tool_calls_made):
            if n == name and a == args_repr:
                count += 1
            else:
                break
        return count

def _args_repr(args: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in sorted(args.items()))

def _print_hop_banner(hop_number: int) -> None:
    print(f"{DIM}{GRAY}── step {hop_number} ──{RESET}")

def run_chat_turn(messages: List[dict[str, Any]]) -> None:
    """
    Run one full user turn, with multiple thinking and tool-call hops.
    """
    stats = TurnStats()
    forced_stop_notice_shown = False

    max_hops = CONFIG.get("max_tool_hops", 20)
    max_repeats = CONFIG.get("max_identical_repeats", 3)

    while True:
        hop_number = stats.hops + 1
        hit_hop_limit = hop_number > max_hops

        thinking = ""
        content = ""
        tool_calls: List[Any] = []
        in_thinking = False
        thinking_finished = False

        active_tools = [] if hit_hop_limit else TOOLS

        if hit_hop_limit and not forced_stop_notice_shown:
            print(
                f"{YELLOW}⚠ Reached the {max_hops}-step tool limit for this "
                f"turn — asking Galileo to wrap up with a final answer.{RESET}\n"
            )
            forced_stop_notice_shown = True

        stream = chat(
            model=CONFIG.get("model", "qwen3:0.6b"),
            messages=messages,
            tools=active_tools,
            think=True,
            stream=True,
        )

        for chunk in stream:
            msg = chunk.message
            if msg.thinking:
                if not in_thinking:
                    in_thinking = True
                    print(f"{BOLD_GRAY}Thinking...{RESET}")
                    print(GRAY, end="", flush=True)
                print(msg.thinking, end="", flush=True)
                thinking += msg.thinking
            if msg.content:
                if in_thinking and not thinking_finished:
                    print(RESET)
                    print(f"{BOLD_GRAY}Done!{RESET}\n")
                    thinking_finished = True
                    in_thinking = False
                    print(f"{CYAN}Galileo:{RESET} ", end="", flush=True)
                print(msg.content, end="", flush=True)
                content += msg.content
            if msg.tool_calls:
                tool_calls.extend(msg.tool_calls)

        if in_thinking and not thinking_finished:
            print(RESET)
            print(f"{BOLD_GRAY}Done!{RESET}\n")

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
            "thinking": thinking or None,
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls

        messages.append(assistant_msg)

        # Execute tool calls if any
        if tool_calls:
            blocked_repeat = False

            for tool_call in tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments or {}
                args_str = _args_repr(args)

                repeat_count = stats.identical_repeat_count(name, args_str)
                if repeat_count >= max_repeats:
                    result = (
                        f"Error: '{name}' has already been called with these exact "
                        f"arguments {max_repeats} times in a row. Stop "
                        f"repeating this call — either use the result you already "
                        f"have, try different arguments, or give your final answer."
                    )

                    print(f"\n{BOLD}🔧 Calling tool:{RESET} {name}({args_str})")
                    print(f"   {RED}→ blocked: identical call repeated too many times{RESET}")
                    blocked_repeat = True
                else:
                    print(f"\n{BOLD}🔧 Calling tool:{RESET} {name}({args_str})")
                    func = AVAILABLE_FUNCTIONS.get(name)
                    if func:
                        try:
                            result = func(**args)
                        except TypeError as e:
                            result = f"Tool error: invalid arguments for {name}: {e}"
                        except Exception as e:
                            result = f"Tool error: {e}"
                    else:
                        result = f"Unknown tool: {name}"

                    is_error = isinstance(result, str) and result.startswith(
                        ("Error", "Tool error", "Unknown tool")
                    )
                    preview = str(result)
                    preview_chars = CONFIG.get("console_preview_chars", 300)
                    if len(preview) > preview_chars:
                        preview = preview[:preview_chars] + "..."
                    color = RED if is_error else ""
                    print(f"   {color}→ {preview}{RESET if color else ''}")

                stats.record(name, args_str)
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": str(result),
                })

            if hit_hop_limit:
                print()
                return

            print()
            if blocked_repeat:
                _print_hop_banner(hop_number + 1)
            continue   # back to the model

        # No tool calls
        if content and content.strip():
            # Final answer
            print()
            return

        # Only thinking (or empty) – continue the loop
        print()
        continue

def main() -> None:
    print("Galileo — agentic Ollama chatbot with persistent memory")
    print(f"Model: {CONFIG.get('model')}  |  Max tool hops: {CONFIG.get('max_tool_hops')}")
    print("Type your message and press Enter.")
    print("Commands: quit / exit / bye\n")

    messages: List[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("You: ").strip().replace('/think', '')
        except (EOFError, KeyboardInterrupt):
            print("\n\nGalileo: Farewell, explorer.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nGalileo: Until next time. Keep questioning everything.")
            break

        messages.append({"role": "user", "content": user_input})
        try:
            run_chat_turn(messages)
        except Exception as e:
            print(f"\n[Error] {e}")
            print("Make sure Ollama is running and the model is pulled:")
            print(f"  ollama pull {CONFIG.get('model')}")
            if messages and messages[-1].get("role") == "user":
                messages.pop()

if __name__ == "__main__":
    main()