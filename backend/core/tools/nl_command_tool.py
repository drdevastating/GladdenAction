"""
core/tools/nl_command_tool.py  — v2  (Git · SQL · Regex · System extended)

Converts plain-English descriptions into shell commands, Git one-liners,
SQL queries, regex patterns, or system-management commands using the Groq
LLM, then executes them through the existing ShellTool safety whitelist.

New command categories in v2
-----------------------------
  git       — git log, diff, stash, rebase, bisect, worktree, reflog, etc.
  sql       — SELECT / INSERT / UPDATE / DELETE / DDL with explanations
  regex     — generate / explain regular expressions (with test strings)
  system    — extended: netstat, tasklist, wmic, powershell pipeline, etc.
  docker    — docker ps, logs, exec, inspect (read-only / safe subset)
  network   — ping, curl, nslookup, tracert, ipconfig
  file_ops  — find, dir, attrib, xcopy, robocopy (safe patterns)

Safety
------
  SQL and Regex are returned as text only — never "executed" through the
  OS.  Git mutations (push, force-push, reset --hard, clean -fd) require
  explicit user opt-in via execute=True AND preview_only=False.
  All shell commands still pass through ShellTool's whitelist.

Event contract
--------------
  type: info | status | error
  stage: translating | command_ready | executing | blocked | done
  tool: nl_command/<mode>
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from groq import Groq

from core.tools.base import BaseTool, ToolResult
from core.tools.shell_tool import ShellTool, _ALLOWED_COMMANDS, _BLOCKED_ARG_PATTERNS

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[dict], None]]

_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
#  Command category detection
# ---------------------------------------------------------------------------

_GIT_SIGNALS = (
    "git ", "commit", "branch", "merge", "rebase", "stash", "cherry-pick",
    "bisect", "reflog", "worktree", "submodule", "tag", "remote", "fetch",
    "pull request", "undo commit", "squash", "amend", "reset",
)

_SQL_SIGNALS = (
    "sql", "select ", "query", "table", "database", "join", "where clause",
    "insert", "update", "delete from", "create table", "drop table",
    "index", "foreign key", "group by", "order by", "having", "subquery",
    "stored procedure", "view", "trigger", "transaction",
)

_REGEX_SIGNALS = (
    "regex", "regular expression", "pattern to match", "pattern for",
    "regexp", "capture group", "lookahead", "lookbehind", "match emails",
    "match urls", "match phone", "match dates", "validate", "extract from",
    "parse ", "find pattern",
)

_DOCKER_SIGNALS = (
    "docker", "container", "image", "dockerfile", "compose", "kubernetes",
    "k8s", "pod", "helm",
)

_NETWORK_SIGNALS = (
    "ping ", "traceroute", "tracert", "nslookup", "dns lookup",
    "curl ", "wget ", "http request", "check connectivity",
    "open ports", "firewall", "ssl cert",
)


def _detect_category(request: str) -> str:
    """Return one of: git | sql | regex | docker | network | shell"""
    low = request.lower()
    if any(s in low for s in _SQL_SIGNALS):
        return "sql"
    if any(s in low for s in _REGEX_SIGNALS):
        return "regex"
    if any(s in low for s in _GIT_SIGNALS):
        return "git"
    if any(s in low for s in _DOCKER_SIGNALS):
        return "docker"
    if any(s in low for s in _NETWORK_SIGNALS):
        return "network"
    return "shell"


# ---------------------------------------------------------------------------
#  LLM system prompts — one per category
# ---------------------------------------------------------------------------

_SHELL_SYSTEM_PROMPT = """\
You are an expert Windows command-line engineer. Your ONLY job is to convert
a plain-English description into the best possible Windows shell command.

STRICT RULES:
1. Output ONLY the raw command — no explanation, no markdown, no code fences,
   no preamble, no trailing period.
2. Prefer CMD commands when they are simpler; use PowerShell syntax when CMD
   cannot do the job cleanly (prefix PowerShell-only commands with "powershell ").
3. Never use commands that delete system files, modify the registry with
   destructive intent, or require elevated privileges that wouldn't normally
   be available.
4. If the request is ambiguous, produce the safest, most broadly useful variant.
5. Use only commands from this allowed list (base command must appear here):
   python, python3, pip, pip3, node, npm, npx, yarn, g++, gcc, cc, c++,
   clang, clang++, make, java, javac, mvn, gradle, go, cargo, rustc, dotnet,
   git, ls, dir, cat, type, head, tail, wc, find, which, where, echo, pwd,
   black, isort, flake8, pylint, mypy, eslint, prettier, tsc, pytest, jest,
   mocha, netstat, tasklist, taskkill, ipconfig, ping, curl, wmic, xcopy,
   robocopy, powershell, set, cls, tree, attrib, fc, comp, sfc, chkdsk.
6. For multi-step operations, chain with && or use a single powershell oneliner.
7. Output exactly one line.

EXAMPLES:
  Request: show all files modified today
  Output:  powershell Get-ChildItem -Recurse | Where-Object {$_.LastWriteTime -gt (Get-Date).Date}

  Request: which process is using port 8080
  Output:  netstat -ano | findstr :8080

  Request: count lines in all .py files
  Output:  powershell (Get-ChildItem -Recurse -Filter *.py | Get-Content | Measure-Object -Line).Lines

  Request: list running node processes
  Output:  tasklist /fi "imagename eq node.exe"

  Request: compress folder dist into dist.zip
  Output:  powershell Compress-Archive -Path dist -DestinationPath dist.zip -Force

  Request: show top 10 largest files in current directory
  Output:  powershell Get-ChildItem -Recurse -File | Sort-Object Length -Descending | Select-Object -First 10 Name,Length
"""

_GIT_SYSTEM_PROMPT = """\
You are a Git expert. Convert a plain-English description into the best
possible Git command or sequence of commands.

STRICT RULES:
1. Output ONLY the raw Git command(s) — no explanation, no markdown fences,
   no preamble.
2. If multiple commands are needed, separate them with " && ".
3. Prefer safe, non-destructive variants unless the user explicitly asks for
   force operations (--force, --hard, -f).
4. For destructive commands (reset --hard, clean -fd, push --force), ALWAYS
   add a clear comment prefix: "# DESTRUCTIVE: " before the command.
5. Never fabricate Git subcommands or flags.
6. Output exactly one logical line (or chained with &&).

EXAMPLES:
  Request: undo the last commit but keep my changes
  Output:  git reset --soft HEAD~1

  Request: show a pretty one-line log of last 20 commits with graph
  Output:  git log --oneline --graph --decorate -20

  Request: find which commit introduced a bug in auth.py
  Output:  git log --all -- auth.py

  Request: stash only staged changes
  Output:  git stash push --staged -m "staged-only"

  Request: list all branches sorted by last commit date
  Output:  git branch -r --sort=-committerdate

  Request: squash last 3 commits into one
  Output:  git rebase -i HEAD~3

  Request: find all commits by Alice in the last month
  Output:  git log --author="Alice" --since="1 month ago" --oneline

  Request: cherry-pick a range of commits
  Output:  git cherry-pick A..B

  Request: show what will be pushed before pushing
  Output:  git log @{u}.. --oneline

  Request: clean up all untracked files and directories
  Output:  # DESTRUCTIVE: git clean -fd
"""

_SQL_SYSTEM_PROMPT = """\
You are a SQL expert. Convert a plain-English description into the best
possible SQL query or DDL statement.

STRICT RULES:
1. Output ONLY the SQL — no markdown code fences, no preamble, no explanation.
2. Write ANSI SQL by default; if the user specifies a dialect (MySQL,
   PostgreSQL, SQLite, MSSQL), use that dialect's syntax.
3. Use clear, readable formatting: uppercase keywords, lowercase identifiers,
   indented subqueries.
4. Add brief inline comments (-- comment) when a clause is non-obvious.
5. Never generate DROP DATABASE, TRUNCATE without a WHERE, or any statement
   that could cause catastrophic data loss without adding a
   "-- WARNING: DESTRUCTIVE" comment.
6. For INSERT/UPDATE/DELETE, always include a WHERE clause or a comment
   explaining why there is none.

EXAMPLES:
  Request: find users who signed up in the last 7 days
  Output:
    SELECT id, email, created_at
    FROM users
    WHERE created_at >= NOW() - INTERVAL '7 days'
    ORDER BY created_at DESC;

  Request: count orders grouped by status
  Output:
    SELECT status, COUNT(*) AS order_count
    FROM orders
    GROUP BY status
    ORDER BY order_count DESC;

  Request: find customers who have never placed an order
  Output:
    SELECT c.id, c.email
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    WHERE o.id IS NULL;

  Request: add an index on users.email
  Output:
    CREATE INDEX idx_users_email ON users (email);
"""

_REGEX_SYSTEM_PROMPT = """\
You are a regex expert. Your job is to produce the best regular expression
for the user's request AND explain it concisely.

OUTPUT FORMAT — output exactly this structure, nothing else:
PATTERN: <the regex>
FLAGS: <flags if any, e.g. i, m, g — or "none">
EXPLANATION: <one-sentence plain-English explanation>
TEST_MATCH: <one example string that would match>
TEST_NO_MATCH: <one example string that would NOT match>

RULES:
1. Use standard PCRE syntax (compatible with Python re, JavaScript, etc.).
2. Prefer readable patterns over clever one-liners — use non-capturing groups
   (?:...) unless capture is essential.
3. Anchor patterns appropriately (^ $ \\A \\Z) based on context.
4. If the user asks for a specific language (Python, JS, Go), add a usage
   snippet AFTER the five required fields, prefixed with SNIPPET:.
5. Never output anything before "PATTERN:".

EXAMPLES:
  Request: match a valid email address
  PATTERN: ^[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}$
  FLAGS: i
  EXPLANATION: Matches standard email addresses with local part, @ symbol, domain, and TLD.
  TEST_MATCH: user.name+tag@example.co.uk
  TEST_NO_MATCH: notanemail@

  Request: extract all URLs from text
  PATTERN: https?://(?:www\\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_+.~#?&/=]*)
  FLAGS: g
  EXPLANATION: Matches http and https URLs including paths and query strings.
  TEST_MATCH: https://www.example.com/path?q=1
  TEST_NO_MATCH: ftp://example.com
"""

_DOCKER_SYSTEM_PROMPT = """\
You are a Docker and container expert. Convert a plain-English description
into the best possible Docker command or docker-compose snippet.

STRICT RULES:
1. Output ONLY the raw command — no markdown fences, no preamble.
2. Prefer read-only / inspection commands unless the user explicitly asks for
   mutations (start, stop, remove, pull).
3. For potentially dangerous operations (docker system prune, rm -f, rmi -f),
   add a "# WARNING:" prefix comment.
4. Use --format flags for cleaner output where applicable.
5. Output exactly one line (or chained with &&).

EXAMPLES:
  Request: show all running containers with their ports
  Output:  docker ps --format "table {{.Names}}\\t{{.Ports}}\\t{{.Status}}"

  Request: tail logs for the web container
  Output:  docker logs -f --tail 100 web

  Request: get a shell inside the api container
  Output:  docker exec -it api /bin/sh

  Request: show disk usage by Docker
  Output:  docker system df -v

  Request: inspect the network of a container
  Output:  docker inspect --format='{{json .NetworkSettings.Networks}}' <container>
"""

_NETWORK_SYSTEM_PROMPT = """\
You are a network diagnostics expert. Convert a plain-English description
into the best possible network command.

STRICT RULES:
1. Output ONLY the raw command — no markdown fences, no preamble.
2. Use Windows-native commands (ping, tracert, nslookup, netstat, ipconfig,
   curl) by default; use PowerShell equivalents when they're cleaner.
3. Never produce commands that modify firewall rules, routing tables, or
   network interfaces without a "# WARNING:" prefix.
4. Output exactly one line.

EXAMPLES:
  Request: check if google.com is reachable
  Output:  ping -n 4 google.com

  Request: trace the route to 8.8.8.8
  Output:  tracert 8.8.8.8

  Request: find my public IP address
  Output:  curl -s https://api.ipify.org

  Request: show all listening ports
  Output:  netstat -an | findstr LISTENING

  Request: flush the DNS cache
  Output:  ipconfig /flushdns
"""


# ---------------------------------------------------------------------------
#  Prompt selector
# ---------------------------------------------------------------------------

_PROMPTS = {
    "shell":   _SHELL_SYSTEM_PROMPT,
    "git":     _GIT_SYSTEM_PROMPT,
    "sql":     _SQL_SYSTEM_PROMPT,
    "regex":   _REGEX_SYSTEM_PROMPT,
    "docker":  _DOCKER_SYSTEM_PROMPT,
    "network": _NETWORK_SYSTEM_PROMPT,
}

# Categories that produce text output (not executed as OS commands)
_TEXT_ONLY_CATEGORIES = {"sql", "regex"}

# Categories that are executable but go through the shell whitelist
_EXECUTABLE_CATEGORIES = {"shell", "git", "docker", "network"}

# Git subcommands that are always safe to run
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "branch", "show", "ls-files", "rev-parse",
    "describe", "shortlog", "remote", "fetch", "stash", "tag",
    "worktree", "submodule", "bisect", "blame", "grep",
})

# Git subcommands that mutate — require explicit execute=True
_MUTATING_GIT_SUBCOMMANDS = frozenset({
    "commit", "push", "pull", "merge", "rebase", "reset", "clean",
    "cherry-pick", "revert", "checkout", "switch", "restore", "rm", "mv",
})


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event(type_: str, stage: str, message: str, category: str = "shell") -> dict:
    return {
        "type":      type_,
        "stage":     stage,
        "message":   message,
        "tool":      f"nl_command/{category}",
        "timestamp": _utc_now(),
    }


def _emit(cb: EventCallback, ev: dict) -> None:
    if cb is None:
        return
    try:
        cb(ev)
    except Exception as exc:  # noqa: BLE001
        logger.warning("event_callback raised: %s", exc)


def _check_shell_safety(command: str) -> tuple[bool, str]:
    """Run ShellTool safety checks on the translated command."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()

    if not parts:
        return False, "Empty command."

    base = parts[0].lower().rstrip(".exe")

    _EXTRA_SAFE = frozenset({
        "netstat", "tasklist", "taskkill", "ipconfig", "ping", "curl",
        "wmic", "xcopy", "robocopy", "set", "cls", "tree", "attrib",
        "fc", "comp", "sfc", "chkdsk", "docker", "nslookup", "tracert",
    })

    if base not in _ALLOWED_COMMANDS and base not in _EXTRA_SAFE:
        if base not in ("powershell", "pwsh"):
            return False, (
                f"Base command '{base}' is not in the allowed list."
            )

    full_lower = command.lower()
    for bad in _BLOCKED_ARG_PATTERNS:
        if bad in full_lower:
            return False, f"Command contains blocked pattern: '{bad}'"

    return True, ""


def _check_git_safety(command: str, execute: bool) -> tuple[bool, str]:
    """
    Validate a Git command.
    Mutating subcommands require execute=True to proceed.
    """
    parts = command.strip().split()
    if not parts or parts[0].lower() not in ("git", "#"):
        return True, ""  # not a plain git command — let shell check handle it

    if parts[0] == "#":  # comment prefix we add for destructive commands
        if not execute:
            return False, "Destructive Git command — set execute=True to confirm."
        return True, ""

    sub = parts[1].lower() if len(parts) > 1 else ""
    if sub in _MUTATING_GIT_SUBCOMMANDS and not execute:
        return False, (
            f"git {sub} modifies the repository. Pass execute=True to run it, "
            "or use preview_only=True to just see the command."
        )

    return True, ""


def _strip_fences(text: str) -> str:
    """Remove ```lang ... ``` fences the model might accidentally add."""
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text).strip()
    return text


def _format_regex_output(raw: str) -> dict:
    """
    Parse the structured regex output into a dict for display.
    Returns dict with keys: pattern, flags, explanation, test_match, test_no_match, snippet.
    """
    result: dict[str, str] = {}
    for line in raw.splitlines():
        for key in ("PATTERN", "FLAGS", "EXPLANATION", "TEST_MATCH", "TEST_NO_MATCH", "SNIPPET"):
            if line.startswith(f"{key}:"):
                result[key.lower()] = line[len(key) + 1:].strip()
    return result


# ---------------------------------------------------------------------------
#  NLCommandTool v2
# ---------------------------------------------------------------------------


class NLCommandTool(BaseTool):
    """
    Natural Language → Command converter + executor.

    Supports: shell commands, Git, SQL, regex, Docker, network diagnostics.
    SQL and regex are returned as text only (never executed).
    All executable commands pass through ShellTool's safety layer.
    """

    name = "nl_command"

    description = (
        "Convert plain-English descriptions into shell commands, Git commands, "
        "SQL queries, regular expressions, Docker commands, or network commands — "
        "and optionally execute them. "
        "Use this when the user says things like: "
        "'how do I …', 'give me the command to …', 'run a command that …', "
        "'find files that …', 'check what process is on port …', "
        "'show me disk usage', 'compress folder X', "
        "'what git command reverts a commit', 'write a SQL query to …', "
        "'give me a regex to match emails', 'docker command to tail logs', etc. "
        "Parameters: "
        "'request' — the natural-language description (required); "
        "'execute' — whether to run the command after translating (default true); "
        "'working_dir' — directory to run in (optional); "
        "'preview_only' — if true, show command but don't run it (default false); "
        "'category' — force a category: shell|git|sql|regex|docker|network (auto-detected if omitted)."
    )

    input_schema: dict[str, Any] = {
        "type": "object",
        "required": ["request"],
        "properties": {
            "request": {
                "type": "string",
                "description": "Plain-English description of the command you want.",
            },
            "execute": {
                "type": "boolean",
                "description": "Whether to execute the translated command. Default true.",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for execution (optional).",
            },
            "preview_only": {
                "type": "boolean",
                "description": "If true, translate but do not execute. Default false.",
            },
            "category": {
                "type": "string",
                "enum": ["shell", "git", "sql", "regex", "docker", "network"],
                "description": "Force a specific category (auto-detected if omitted).",
            },
        },
    }

    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        self._client = Groq(api_key=api_key) if api_key else None
        self._shell  = ShellTool()

    # ------------------------------------------------------------------ #

    def execute(self, **kwargs: Any) -> ToolResult:
        cb: EventCallback = kwargs.pop("event_callback", None)

        request      = kwargs.get("request", "").strip()
        do_execute   = kwargs.get("execute", True)
        working_dir  = kwargs.get("working_dir", "")
        preview_only = kwargs.get("preview_only", False)
        forced_cat   = kwargs.get("category", "").strip().lower()

        if not request:
            return ToolResult(success=False, error="'request' is required.")

        if self._client is None:
            return ToolResult(
                success=False,
                error="GROQ_API_KEY not set — cannot translate command.",
            )

        # ── Detect category ─────────────────────────────────────────────
        category = forced_cat if forced_cat in _PROMPTS else _detect_category(request)

        _emit(cb, _event("info", "translating",
                          f"[{category.upper()}] Translating: {request}…", category))

        # ── Translate ────────────────────────────────────────────────────
        raw = self._translate(request, category)
        if not raw:
            return ToolResult(
                success=False,
                error="LLM did not produce a valid output. Try rephrasing.",
            )

        command = _strip_fences(raw)

        # ── Handle text-only categories (SQL, Regex) ─────────────────────
        if category == "sql":
            return self._handle_sql(command, request, cb, preview_only)

        if category == "regex":
            return self._handle_regex(command, request, cb, preview_only)

        # ── All executable categories ────────────────────────────────────
        _emit(cb, _event("status", "command_ready",
                          f"Translated command: {command}", category))

        # Git-specific safety check
        if category == "git":
            ok, reason = _check_git_safety(command, bool(do_execute))
            if not ok:
                _emit(cb, _event("error", "blocked", f"Git safety: {reason}", category))
                return ToolResult(
                    success=False,
                    error=reason,
                    metadata={"translated_command": command, "category": category},
                )
            # Strip destructive comment prefix before running
            if command.startswith("# DESTRUCTIVE:"):
                command = command[len("# DESTRUCTIVE:"):].strip()

        # General shell safety
        is_safe, reason = _check_shell_safety(command)
        if not is_safe:
            _emit(cb, _event("error", "blocked",
                              f"Command blocked by safety filter: {reason}", category))
            return ToolResult(
                success=False,
                error=f"Command blocked by safety filter: {reason}",
                metadata={"translated_command": command, "blocked_reason": reason,
                          "category": category},
            )

        # ── Preview-only ─────────────────────────────────────────────────
        if preview_only or not do_execute:
            _emit(cb, _event("status", "done",
                              f"Preview only — not executed: {command}", category))
            return ToolResult(
                success=True,
                output=f"Command (not executed): {command}",
                metadata={
                    "translated_command": command,
                    "executed": False,
                    "request": request,
                    "category": category,
                },
            )

        # ── Execute ──────────────────────────────────────────────────────
        _emit(cb, _event("info", "executing", f"Executing: {command}", category))

        shell_kwargs: dict[str, Any] = {"command": command}
        if working_dir:
            shell_kwargs["working_dir"] = working_dir
        if cb is not None:
            shell_kwargs["event_callback"] = cb

        result = self._shell.execute(**shell_kwargs)

        stage = "done" if result.success else "done"
        etype = "status" if result.success else "error"
        _emit(cb, _event(etype, stage,
                          f"Command {'completed' if result.success else 'failed'}: "
                          f"{str(result.output or result.error or '')[:120]}",
                          category))

        meta = dict(result.metadata or {})
        meta["translated_command"] = command
        meta["request"]            = request
        meta["category"]           = category

        return ToolResult(
            success=result.success,
            output=result.output,
            error=result.error,
            metadata=meta,
        )

    # ================================================================== #
    #  Category-specific handlers                                          #
    # ================================================================== #

    def _handle_sql(
        self,
        sql: str,
        request: str,
        cb: EventCallback,
        preview_only: bool,
    ) -> ToolResult:
        """SQL is always returned as text — never executed through the OS."""
        is_destructive = any(kw in sql.upper() for kw in (
            "DROP ", "TRUNCATE ", "DELETE ", "ALTER TABLE", "UPDATE "
        ))
        warning = ""
        if is_destructive:
            warning = "\n\n-- ⚠  Review carefully before running against production data."
            _emit(cb, _event("info", "sql_warning",
                              "SQL contains destructive operation — review before running.",
                              "sql"))

        _emit(cb, _event("status", "command_ready",
                          f"SQL query generated ({len(sql)} chars)", "sql"))

        output = sql + warning
        _emit(cb, _event("status", "done",
                          "SQL query ready — copy and run in your database client.", "sql"))

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "category":        "sql",
                "request":         request,
                "executed":        False,
                "is_destructive":  is_destructive,
                "translated_command": sql,
            },
        )

    def _handle_regex(
        self,
        raw: str,
        request: str,
        cb: EventCallback,
        preview_only: bool,
    ) -> ToolResult:
        """Parse and display a structured regex response."""
        parsed = _format_regex_output(raw)
        pattern = parsed.get("pattern", raw)

        _emit(cb, _event("status", "command_ready",
                          f"Regex pattern: {pattern}", "regex"))

        # Build a rich display output
        lines = [f"PATTERN:        {parsed.get('pattern', '(none)')}"]
        if parsed.get("flags") and parsed["flags"].lower() not in ("none", ""):
            lines.append(f"FLAGS:          {parsed['flags']}")
        if parsed.get("explanation"):
            lines.append(f"EXPLANATION:    {parsed['explanation']}")
        if parsed.get("test_match"):
            lines.append(f"✓ MATCHES:      {parsed['test_match']}")
        if parsed.get("test_no_match"):
            lines.append(f"✗ NO MATCH:     {parsed['test_no_match']}")
        if parsed.get("snippet"):
            lines.append(f"\nUSAGE SNIPPET:\n{parsed['snippet']}")

        output = "\n".join(lines)

        _emit(cb, _event("status", "done",
                          f"Regex ready — pattern: {pattern}", "regex"))

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "category":           "regex",
                "request":            request,
                "executed":           False,
                "pattern":            pattern,
                "flags":              parsed.get("flags", ""),
                "translated_command": pattern,
            },
        )

    # ================================================================== #
    #  LLM translation                                                     #
    # ================================================================== #

    def _translate(self, request: str, category: str) -> str | None:
        system_prompt = _PROMPTS.get(category, _SHELL_SYSTEM_PROMPT)
        try:
            response = self._client.chat.completions.create(
                model=_DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": request},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            text = (response.choices[0].message.content or "").strip()
            return text if text else None
        except Exception as exc:  # noqa: BLE001
            logger.error("NLCommandTool._translate failed: %s", exc)
            return None

    def __repr__(self) -> str:
        return "<NLCommandTool v2>"