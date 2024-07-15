import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate


from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

template = '''You are a helpful assistant. Answer the following question as best you can using the following tools:
{tools_description}
Try to use a tool at least one time
The tools are not an assistant, they are callable tools that require an a script to be used
Call a tool using it name, for example: 'Action: duckduckgo_results_json' and after: 'Action Input: your search here'
Use the exactly following format in this order:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]. Ever use an 'Action' or 'Final Answer' after the 'Thought'!
Action Input: the input to the action. Ever use an 'Action Input' after the 'Action'!
Observation: the result of the action. Ever use an 'Observation' after the 'Action Input'!
... (this Thought/Action/Action Input/Observation can repeat in this exactly order)
Thought: I now know the final answer
Final Answer: the final answer with every helpful information to the original input question.

Never Ask for a tool, instead, use: 'Action: ' and 'Action Input: '

Begin!

Question: {input}
Thought:{agent_scratchpad}'''
prompt = PromptTemplate.from_template(template)

llm = ChatLlamaCpp(
    #model_path="E:/DEV/brincandoDeAi/brincando-de-ai-python/llm/Qwen2-0.5B-Instruct.Q8_0.gguf", #Qwen 2-0 5B Instruct Q6 GUFF
    model_path="E:/DEV/brincandoDeAi/brincando-de-ai-python/llm/Meta-Llama-3-8B-Instruct.Q6_K.gguf", #Meta Llama 3 8B Instruct Q6 GUFF
    #model_path="E:/DEV/brincandoDeAi/brincando-de-ai-python/llm/TinyLlama-1.1B-Chat-v1.0-UltraQ-Imat-NEO1-Q8_0-imat.gguf", #TinyLlama 1 1.1B Chat Q6 GUFF
    max_tokens=700,
    temperature=0,
    top_p=1,
    verbose=True,
    n_ctx=10000
)
#prompt = hub.pull("hwchase17/react")

ddgSearch = DuckDuckGoSearchAPIWrapper()
ddg_search_tool = DuckDuckGoSearchResults(verbose=True, api_wrapper=ddgSearch)

tools = [ddg_search_tool]

question = "Tell me about what happened with Donald Trump in 13/07/2024"

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response = agent_executor.invoke({
    "input": f"{question}",
})

print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
print(response)
print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")