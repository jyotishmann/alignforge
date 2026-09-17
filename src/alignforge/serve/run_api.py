"""Module entry point for the API server — used by serve all subprocess."""

import typer
import uvicorn

from alignforge.serve.app import create_app


def main(port: int = 8000, engine: str = "ollama") -> None:
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)


if __name__ == "__main__":
    typer.run(main)
