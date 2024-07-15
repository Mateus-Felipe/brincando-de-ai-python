from langchain import hub
from typing import Annotated
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AnyMessage
from langchain_community.chat_models import ChatLlamaCpp
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search  import TavilySearchResults


template_prompt = '''
You are a helpful assistant. You are a helpful AI. Answer the following question as best you can. In the end, show all links of the search tool if you used any.
Use a tool at least one time. Call a tool using it name, for example: 'Action: duckduckgo_results_json'. Use the exactly following format in this order:
Question: the input question you must answer
Thought: you should always think about what to do,
Action: the action to take, should be one of your tools,
Action Input: the input to the action,
Observation: the result of the action,
... (this Thought/Action/Action Input/Observation can repeat N times in this exactly order),
Thought: I now know the final answer,
Final Answer: the final answer to the original Human input message
Begin!
'''

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

model = ChatLlamaCpp(
    #model_path="E:/DEV/brincandoDeAi/brincando-de-ai-python/llm/Qwen2-0.5B-Instruct.Q8_0.gguf", #Qwen 2-0 5B Instruct Q6 GUFF
    model_path="./llm/Meta-Llama-3-8B-Instruct.Q6_K.gguf", #Meta Llama 3 8B Instruct Q6 GUFF
    #model_path="E:/DEV/brincandoDeAi/brincando-de-ai-python/llm/TinyLlama-1.1B-Chat-v1.0-UltraQ-Imat-NEO1-Q8_0-imat.gguf", #TinyLlama 1 1.1B Chat Q6 GUFF
    temperature=0.1,
    top_p=1,
    verbose=True,
    n_ctx=10000
)
ddgSearch = DuckDuckGoSearchAPIWrapper()
ddg_search_tool = DuckDuckGoSearchRun(verbose=True, api_wrapper=ddgSearch)
search = TavilySearchAPIWrapper(tavily_api_key="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")
search_tool = TavilySearchResults(api_wrapper=search, max_results=2)
tools = [search_tool, ddg_search_tool]


model_tools = model.bind_tools(tools)
question = "Tell me about what happened with Donald Trump in july 13rd 2024"

agent = create_react_agent(model=model_tools, tools=tools, debug=True, messages_modifier=template_prompt)

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

response = agent.invoke(
    {
    "messages": [
    SystemMessage(content="You are a helpful assistant. You are a helpful AI. Answer the following question as best you can. In the end, show all links of the search tool if you used any."),
    HumanMessage(content=question)
    ]}
)

print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
print(response)


# print_stream(agent.stream({"messages": [
#     SystemMessage(content="You are a helpful assistant. You are a helpful AI. Answer the following question as best you can. In the end, show all links of the search tool if you used any."),
#     AIMessage(content="You are a helpful AI. Answer the following question as best you can. In the end, show all links of the search tool if you used any."),
#     HumanMessage(content=question)
#     ]
# }, stream_mode="values"))

'''
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response = agent_executor.invoke({
    "input": f"{question}",
})

print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
print(response)
print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
'''