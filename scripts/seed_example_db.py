"""Seed a minimal registry database for the demo. Run once; commit the output."""

from __future__ import annotations

from pathlib import Path

from alignforge.core.registry import Registry


def main() -> None:
    db_path = Path("registry.example.db")
    if db_path.exists():
        db_path.unlink()

    reg = Registry(db_path)
    reg.connect()

    # Pre-register the three comparison models backed by EchoEngine.
    reg.publish_model("base", "Base (Qwen2.5-1.5B)", "echo", "echo", sort_order=1)
    reg.publish_model("sft", "SFT (+QLoRA)", "echo", "echo", sort_order=2)
    reg.publish_model("dpo", "DPO (β=0.1)", "echo", "echo", sort_order=3)

    print(f"Seeded {db_path} with 3 echo models.")
    reg.close()


if __name__ == "__main__":
    main()
