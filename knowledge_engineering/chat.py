"""Interactive terminal interface for the KAIRIX Investigation Agent."""
from __future__ import annotations

from .rag_service import RAGService


def main() -> None:
    print("KAIRIX Investigation Agent")
    print("Type 'exit' or 'quit' to stop.\n")

    service = RAGService()
    service.connect()
    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            try:
                result = service.investigate(question)
            except Exception as exc:
                print(f"Agent error: {exc}\n")
                continue

            print(f"\nAnswer: {result['answer']}")
            print(f"Status: {result.get('status', 'SUCCESS')}")
            print(f"Confidence: {result['confidence']:.2f} ({result['confidence_level']})")
            print(f"Investigation triggered: {result['investigation_triggered']}")

            gaps = result.get("knowledge_gaps", [])
            if gaps:
                print("Knowledge gaps:")
                for gap in gaps:
                    print(f"- {gap}")

            evidence_ids = result.get("evidence_ids", [])
            if evidence_ids:
                print(f"Evidence IDs: {', '.join(evidence_ids)}")
            print()
    finally:
        service.close()


if __name__ == "__main__":
    main()
