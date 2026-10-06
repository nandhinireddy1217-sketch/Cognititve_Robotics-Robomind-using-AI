import ollama
from rag.knowledge_base import search_knowledge


def ask_robomind(question):

    # Search ChromaDB
    documents = search_knowledge(question)

    # Combine retrieved information
    context = "\n\n".join(documents)

    prompt = f"""
You are RoboMind, an intelligent cognitive robot.

Use the following information from the RoboMind knowledge base
to answer the user's question.

KNOWLEDGE BASE:
{context}

USER QUESTION:
{question}

Answer clearly and simply.
If the answer is not available in the knowledge base,
say that the information is not available in the knowledge base.
"""

    # Ask Llama 3.2
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":

    question = input("Ask RoboMind: ")

    answer = ask_robomind(question)

    print("\nRoboMind:")
    print(answer)