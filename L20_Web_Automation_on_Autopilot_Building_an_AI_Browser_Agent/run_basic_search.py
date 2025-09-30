import os
import asyncio
from dotenv import load_dotenv
from browser_use import Agent, ChatGoogle


load_dotenv()


async def main():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to .env or environment "
            "variables."
        )

    llm = ChatGoogle(model="gemini-2.5-flash")
    task = (
        "Search Google for 'what is browser automation' and tell me the top 3 "
        "results"
    )
    agent = Agent(task=task, llm=llm)
    history = await agent.run()

    urls = history.urls()
    print("Visited URLs:")
    for url in urls:
        print(url)

    with open("visited_urls.txt", "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print("\nSaved to visited_urls.txt")


if __name__ == "__main__":
    asyncio.run(main())
