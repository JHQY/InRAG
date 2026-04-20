import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test: ensure table hits appear in retrieve_context() output."
    )
    parser.add_argument(
        "--query",
        default="表格",
        help="Query string to run against the retriever.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to retrieve.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild the index before testing.",
    )
    args = parser.parse_args()

    if args.reindex:
        from ingestion.indexer import build_index

        print("Rebuilding index...")
        build_index()

    from retrieval.retriever import RAGInterface

    rag = RAGInterface()
    results = rag.retrieve(args.query, top_k=args.top_k)

    print("\n=== RETRIEVE RESULTS ===")
    if not results:
        print("No results.")
    else:
        for i, r in enumerate(results, 1):
            print(f"{i}. modality={r.get('modality')} score={r.get('score')}")
            text = r.get("text") or ""
            if text:
                print(f"   text: {text[:200]}")
            table = r.get("table") or {}
            if table:
                print("   table: " + json.dumps(table, ensure_ascii=False)[:200])

    print("\n=== RETRIEVE CONTEXT ===")
    context = rag.retrieve_context(args.query, top_k=args.top_k)
    if not context:
        print("Context is empty.")
    else:
        print(context[:2000])


if __name__ == "__main__":
    main()
