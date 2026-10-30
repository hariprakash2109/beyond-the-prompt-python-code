"""Command-line interface: ingest documents, ask questions, chat with the agent.

    python -m knowledge_assistant.cli ingest
    python -m knowledge_assistant.cli ask "How long is an internship?" --role student
    python -m knowledge_assistant.cli chat --user alice --role student
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

from dotenv import load_dotenv

from .access import LEVELS, User
from .config import ConfigError, Settings
from .embeddings import build_embedder
from .services import Services, build_services, ingest


def interactive_chat(services: Services, user: User, input_fn: Callable[[str], str] = input,
                     output_fn: Callable[[str], None] = print) -> None:
    output_fn(f"Chatting as {user.name} ({user.role}). Type 'quit' to exit.")
    while True:
        try:
            question = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if question.lower() == "quit":
            return
        if len(question) < 3:
            continue
        result = services.agent.run(question, user)
        while result.status == "needs_approval":
            action = result.pending
            output_fn(f"[approval needed] The assistant wants to run '{action.tool_name}' with {action.arguments}")
            decision = input_fn("approve? [y/N] ").strip().lower() == "y"
            result = services.agent.resume(action.pending_id, decision, user)
        if result.status == "answered":
            output_fn(f"ai> {result.answer}")
        else:
            output_fn(f"[stopped: {result.stop_reason}] No final answer was produced.")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(prog="knowledge_assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Build the search index from KB_DATA_DIR")
    for name in ("ask", "chat"):
        p = sub.add_parser(name)
        p.add_argument("--user", default="cli-user")
        p.add_argument("--role", default="student", choices=sorted(LEVELS))
        if name == "ask":
            p.add_argument("question")
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
        if args.command == "ingest":
            store = ingest(settings, build_embedder(settings))
            print(f"Indexed {len(store)} chunks into {settings.index_path}")
            return 0
        services = build_services(settings)
        user = User(args.user, args.role)
        if args.command == "ask":
            result = services.rag.ask(args.question, user)
            print(result.answer)
            for c in result.citations:
                print(f"  - {c.title} ({c.source}) score={c.score:.2f}")
            return 0
        interactive_chat(services, user)
        return 0
    except ConfigError as exc:
        print(f"Configuration problem: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
