# taxonomy.py
"""
Dynamic taxonomy mapping.

Parses `taxonomy.md` (located in the same repository) and builds a dictionary
`TAXONOMY_MAP` that maps YOLO class names to the taxonomy categories defined in
the markdown file.

The expected markdown format is simple headings for categories followed by a
list of class names, e.g.:

```markdown
## Vehicle
- car
- truck
## VulnerableRoadUser
- pedestrian
- cyclist
```

If the markdown structure differs, the parser will skip lines it cannot
interpret.
"""

import pathlib
import re

# Resolve the path to taxonomy.md relative to this file
_TAXONOMY_MD_PATH = pathlib.Path(__file__).with_name("taxonomy.md")

def _parse_taxonomy() -> dict:
    """Parse `taxonomy.md` and return a mapping of class name → category.

    Returns:
        dict: Mapping where keys are YOLO class names (as strings) and values are
        the taxonomy category (also a string).
    """
    mapping: dict[str, str] = {}
    current_category: str | None = None
    for line in _TAXONOMY_MD_PATH.read_text().splitlines():
        # Detect top‑level category headings (e.g., "## Vehicle")
        cat_match = re.match(r"^##\s+(.*)", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            continue
        # Detect list items (e.g., "- car")
        item_match = re.match(r"^-\s*(\w+)", line)
        if item_match and current_category:
            class_name = item_match.group(1).strip()
            mapping[class_name] = current_category
    return mapping

# Build the mapping once at import time – this is cheap for a small file.
TAXONOMY_MAP: dict[str, str] = _parse_taxonomy()

def get_taxonomy(class_name: str) -> str:
    """Return the taxonomy category for *class_name* or "Unknown" if not found."""
    return TAXONOMY_MAP.get(class_name, "Unknown")
