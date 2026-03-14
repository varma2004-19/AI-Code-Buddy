from logic import process_query
from dotenv import load_dotenv
load_dotenv()

def main():
    print("\nAI Code Coach")
    print("1) Debug Code")
    print("2) Translate Code")
    print("3) Explain Algorithm")

    choice = input("Choose (1/2/3): ").strip()
    query = input("Describe your problem: ")

    result, docs, fix_info = process_query(choice, query)

    print("\n--- AI Response ---\n")
    print(result)

    print("\n--- Relevant Code ---")
    sources = set(doc.metadata.get('source', 'Unknown') for doc in docs)
    for src in sources:
        print(f"- {src}")

if __name__ == "__main__":
    main()