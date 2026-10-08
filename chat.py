"""Run an interactive text session or a reproducible one-shot response."""
import argparse
import json
from src.conversation import ConversationSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["ppo", "rule"], default="ppo")
    parser.add_argument("--generator", choices=["template", "ollama", "hybrid"], default="template")
    parser.add_argument("--age", type=int)
    parser.add_argument("--budget", choices=["unknown", "low", "mid", "high"], default="unknown")
    parser.add_argument("--message", help="Print one structured response and exit")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    session = ConversationSession(args.policy, args.generator, age=args.age, budget=args.budget)
    if args.message:
        print(json.dumps(session.reply(args.message), indent=2))
        return
    print("Insurance research conversation. Products are synthetic. Type /quit to exit.")
    while not session.closed:
        try:
            text = input("Customer: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if not text:
            continue
        result = session.reply(text)
        print("Agent: " + result["text"])
        if args.debug:
            print(json.dumps({"action": result["action"], "strategy": result["strategy"],
                              "understanding": result["understanding"], "generation_source": result["source"],
                              "llm_error": result["llm_error"], "llm_rejected": result["llm_rejected"],
                              "llm_rejection_reasons": result["llm_rejection_reasons"],
                              "decision_source": result["decision_source"], "decision_reason": result["decision_reason"]}))


if __name__ == "__main__":
    main()
