from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_to_txt

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatAnthropic(model_name="claude-sonnet-5-5")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper, 
            Answer the user query and use neccessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can I help you research?")
raw_response = agent_executor.invoke({"query": query})

output = raw_response.get("output")

if isinstance(output, str):
    text = output
else:
    text = ""
    for block in output:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block["text"]

try:
    structured_response = parser.parse(text)
    print(structured_response)
    save_to_txt(structured_response.model_dump_json(indent=2))
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)