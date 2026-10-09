"""MCP (Model Context Protocol) server adapter (optional).

Exposes ``impute_table`` as an MCP tool. Like the langchain adapter, this is
dependency-free: it returns a plain tool descriptor that an MCP host can mount.
"""


def mcp_tool_spec(engine):
    """Return an MCP tool descriptor for the imputation capability."""
    return {
        "name": "impute_table",
        "description": "Fill missing cells (null) in a table using a tabular foundation model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "records": {"type": "array", "items": {"type": "object"}},
                "columns": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["records"],
        },
        "handler": engine.impute_records,
    }
