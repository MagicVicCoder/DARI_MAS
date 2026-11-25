"""Conservative syntax repair; never complete truncated or missing model content."""
import json
import re


def repair_json(text):
    try:
        json.loads(text)
        return text, []
    except ValueError:
        pass
    original = text
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", stripped, re.DOTALL)
    changes = []
    if fenced:
        text = fenced.group(1)
        changes.append("whole_document_json_fence_removed")
    chunks = []
    inside = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            inside = not inside
        if inside and char == "\\" and index+1 < len(text):
            following = text[index+1]
            if following not in '"\\/bfnrtu':
                chunks.append("\\")
                changes.append("illegal_json_escape_preserved_as_literal_backslash")
            else:
                chunks.append(char)
                chunks.append(following)
                index += 2
                continue
        chunks.append(char)
        index += 1
    repaired = "".join(chunks)
    try:
        json.loads(repaired)
    except ValueError:
        return original, []
    return repaired, sorted(set(changes))
