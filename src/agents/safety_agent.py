"""SQL safety validator.

Code-based validation - intentionally strict so the LLM cannot escape
read-only execution. The validator:

* trims input
* requires the statement to start with SELECT or WITH
* rejects any blocked DDL/DML keyword
* rejects multiple statements (more than one ``;`` separator)
* validates that every table mentioned exists in ``src/schema.py``
* runs the query through ``sqlparse`` to confirm it parses
* appends ``LIMIT 100`` when the user did not provide one
"""
from __future__ import annotations

import re
from typing import Any

import sqlparse
from sqlparse.sql import Identifier, IdentifierList, Token, TokenList
from sqlparse.tokens import DML, Keyword

from src.schema import is_known_table


BLOCKED_KEYWORDS: tuple[str, ...] = (
    "drop",
    "delete",
    "insert",
    "update",
    "alter",
    "create",
    "truncate",
    "replace",
    "attach",
    "detach",
    "copy",
    "pragma",
    "call",
    "export",
    "grant",
    "revoke",
)


def _strip_trailing_semicolons(sql: str) -> str:
    """Remove trailing whitespace and semicolons (we re-add at most one)."""
    return sql.strip().rstrip(";").strip()


def _count_statements(sql: str) -> int:
    statements = [
        s for s in sqlparse.split(sql) if s.strip() and s.strip() != ";"
    ]
    return len(statements)


def _has_blocked_keyword(sql_lower: str) -> str | None:
    for kw in BLOCKED_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", sql_lower):
            return kw
    return None


def _extract_identifier_strings(token: Token) -> list[str]:
    """Recursively pull table-like identifier strings from a token tree."""
    names: list[str] = []
    if isinstance(token, IdentifierList):
        for ident in token.get_identifiers():
            names.extend(_extract_identifier_strings(ident))
    elif isinstance(token, Identifier):
        # ``schema.table alias`` - we want the real_name.
        real = token.get_real_name()
        if real:
            names.append(real)
    elif isinstance(token, TokenList):
        for sub in token.tokens:
            names.extend(_extract_identifier_strings(sub))
    return names


def _collect_tables(parsed: TokenList) -> set[str]:
    """Walk the parse tree pulling out table names after FROM and JOIN."""
    tables: set[str] = set()
    structured = list(parsed.tokens)

    def walk(token_list: list[Token]) -> None:
        i = 0
        while i < len(token_list):
            token = token_list[i]
            if token.ttype is Keyword and token.normalized.upper() in (
                "FROM",
                "JOIN",
                "INNER JOIN",
                "LEFT JOIN",
                "RIGHT JOIN",
                "FULL JOIN",
                "OUTER JOIN",
                "CROSS JOIN",
                "LEFT OUTER JOIN",
                "RIGHT OUTER JOIN",
                "FULL OUTER JOIN",
            ):
                j = i + 1
                while j < len(token_list) and (token_list[j].is_whitespace or token_list[j].ttype is sqlparse.tokens.Comment):
                    j += 1
                if j < len(token_list):
                    nxt = token_list[j]
                    for name in _extract_identifier_strings(nxt):
                        tables.add(name)
                i = j
            elif isinstance(token, TokenList):
                walk(list(token.tokens))
            i += 1

    walk(structured)
    return tables


_CTE_NAME_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s*\(", re.IGNORECASE)


def _collect_cte_names(sql: str) -> set[str]:
    """Best-effort: pull CTE aliases out of any WITH ... AS ( ... ) clauses."""
    if "with" not in sql.lower():
        return set()
    return {m.group(1) for m in _CTE_NAME_RE.finditer(sql)}


def _ensure_limit(sql: str, default_limit: int = 100) -> str:
    """Append a LIMIT clause if none is present."""
    no_semi = _strip_trailing_semicolons(sql)
    lower = no_semi.lower()
    # Quick check - does the statement already have a LIMIT keyword?
    if re.search(r"\blimit\b", lower):
        return no_semi + ";"
    return f"{no_semi}\nLIMIT {default_limit};"


def safety_agent(sql: str) -> dict[str, Any]:
    """Validate SQL and return ``{approved, reason, safe_sql}``."""
    if not sql or not sql.strip():
        return {"approved": False, "reason": "Empty SQL.", "safe_sql": ""}

    cleaned = sql.strip()
    no_semi = _strip_trailing_semicolons(cleaned)
    lower = no_semi.lower()

    # Reject multiple statements.
    if _count_statements(cleaned) > 1:
        return {
            "approved": False,
            "reason": "Multiple SQL statements are not allowed.",
            "safe_sql": "",
        }

    # Must start with SELECT or WITH.
    leading = re.match(r"^\s*(\w+)", no_semi)
    if not leading or leading.group(1).lower() not in ("select", "with"):
        return {
            "approved": False,
            "reason": "Only SELECT and WITH statements are allowed.",
            "safe_sql": "",
        }

    # Block DDL/DML keywords.
    blocked = _has_blocked_keyword(lower)
    if blocked:
        return {
            "approved": False,
            "reason": f"Blocked keyword detected: {blocked.upper()}",
            "safe_sql": "",
        }

    # Parse with sqlparse.
    try:
        parsed_statements = sqlparse.parse(cleaned)
    except Exception as exc:
        return {
            "approved": False,
            "reason": f"SQL could not be parsed: {exc}",
            "safe_sql": "",
        }
    if not parsed_statements:
        return {"approved": False, "reason": "SQL could not be parsed.", "safe_sql": ""}
    parsed = parsed_statements[0]

    # Confirm the parser detected a SELECT statement.
    first_dml = next((t for t in parsed.flatten() if t.ttype is DML), None)
    if first_dml is not None and first_dml.value.upper() not in ("SELECT",):
        return {
            "approved": False,
            "reason": f"Disallowed DML detected: {first_dml.value.upper()}",
            "safe_sql": "",
        }

    # Validate referenced tables (CTE aliases declared in WITH ... are OK).
    tables = _collect_tables(parsed)
    cte_names = _collect_cte_names(cleaned)
    unknown = sorted(t for t in tables if not is_known_table(t) and t not in cte_names)
    if unknown:
        return {
            "approved": False,
            "reason": f"Unknown table(s): {', '.join(unknown)}",
            "safe_sql": "",
        }

    safe_sql = _ensure_limit(no_semi)
    return {"approved": True, "reason": "OK", "safe_sql": safe_sql}
