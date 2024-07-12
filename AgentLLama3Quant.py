import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

from langchain_community.tools import DuckDuckGoSearchRun
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import LlamaCpp

os.environ["TAVILY_API_KEY"] = "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"


template = """You are a helpfull agent AI.
Question: {question}.
Answer: Let's work this out in a step by step way to be sure we have the right answer."""

llm = LlamaCpp(
    model_path="C:/Users/mateus.ramos/models/llm/Qwen2-0.5B-Instruct.Q8_0.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    verbose=True,
    n_ctx=10000
)
prompt = hub.pull("hwchase17/react")

ddgSearch = DuckDuckGoSearchAPIWrapper()
ddg_search_tool = DuckDuckGoSearchRun(verbose=True)

searchTavily = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=searchTavily)
tools = [ddg_search_tool]
# tools = [tavily_tool]

question = """Como um brasileiro pode se mudar para a Finlândia?"""

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response = agent_executor.invoke({
    "input": f"{question} Only use a tool if needed, otherwise respond with Final Answer. If you got the response from the tool, continue for the next step.",
})

print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
print(response)
print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")


# def __main__():
