"""LangChain integration (optional).

Exposes the imputation capability as a LangChain ``BaseTool`` if langchain is
installed; otherwise the class is importable but inactive. Keeping this optional
means the core framework stays zero-dependency.
"""


class ImputeTool:
    """A thin, framework-agnostic tool wrapper around a TabularEngine.

    This deliberately does NOT import langchain at module scope so the core
    library has no hard dependency. If langchain is available, you can wrap it
    with ``StructuredTool.from_function(ImputeTool(engine).run)``.
    """

    name = "impute_table"
    description = "Fill missing cells (null) in a table using a tabular foundation model."

    def __init__(self, engine):
        self.engine = engine

    def run(self, records, columns=None):
        return self.engine.impute_records(records, columns)

    def __call__(self, records, columns=None):
        return self.run(records, columns)
