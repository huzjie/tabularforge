"""Minimal YAML subset parser (zero-dependency fallback).

Supports what the project configs actually use:
- ``key: value`` scalars (str / int / float / bool / null)
- inline comments (``key: value  # comment``)
- block lists (``key:\n  - a\n  - b``)
- nested maps via indentation
- inline lists ``[1, 2, 3]``

The parser keeps a stack of ``(indent, container, parent, key)`` so block lists
are materialized in-place inside their parent map. This avoids the classic bug
where the first list item gets dropped.
"""


def _scalar(text):
    t = text.strip()
    if t == "" or t == "null" or t == "~":
        return None
    if t == "true":
        return True
    if t == "false":
        return False
    low = t.lower()
    if low in ("yes",):
        return True
    if low in ("no",):
        return False
    if t.startswith("[") and t.endswith("]"):
        inner = t[1:-1].strip()
        if inner == "":
            return []
        return [_scalar(x) for x in inner.split(",")]
    if t.startswith('"') and t.endswith('"'):
        return t[1:-1]
    if t.startswith("'") and t.endswith("'"):
        return t[1:-1]
    try:
        if "." in t or "e" in t.lower():
            return float(t)
        return int(t)
    except ValueError:
        return t


def parse(text):
    root = {}
    stack = [( -1, root, None, None)]  # (indent, container, parent, key)
    for raw in text.splitlines():
        if raw.strip() == "" or raw.lstrip().startswith("#"):
            continue
        # strip inline comment (not inside quotes)
        line = _strip_comment(raw)
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- "):
            item = _scalar(stripped[2:])
            _handle_list_item(stack, indent, item)
        elif stripped.startswith("-"):
            item = _scalar(stripped[1:])
            _handle_list_item(stack, indent, item)
        else:
            if ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            _handle_map_entry(stack, indent, key, val)
    return root


def _strip_comment(line):
    in_s = False
    in_d = False
    for i, ch in enumerate(line):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d:
            return line[:i]
    return line


def _pop_to(stack, indent):
    while len(stack) > 1 and stack[-1][0] >= indent:
        stack.pop()


def _handle_map_entry(stack, indent, key, val):
    _pop_to(stack, indent)
    parent = stack[-1][1]
    if val == "":
        child = {}
        if isinstance(parent, list):
            parent.append({key: child})
            stack.append((indent, child, parent, key))
        else:
            parent[key] = child
            stack.append((indent, child, parent, key))
    else:
        v = _scalar(val)
        if isinstance(parent, list):
            parent.append({key: v})
        else:
            parent[key] = v


def _handle_list_item(stack, indent, item):
    _pop_to(stack, indent)
    parent = stack[-1][1]
    if not isinstance(parent, list):
        # replace the placeholder dict with a list
        grandparent = stack[-1][2]
        key = stack[-1][3]
        lst = []
        if grandparent is None:
            return
        if isinstance(grandparent, list):
            grandparent[-1][key] = lst
        else:
            grandparent[key] = lst
        stack[-1] = (stack[-1][0], lst, grandparent, key)
        parent = lst
    if isinstance(item, dict):
        parent.append(item)
        stack.append((indent, item, parent, None))
    else:
        parent.append(item)
