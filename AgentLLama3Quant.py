import os
import getpass
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import PromptTemplate

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(
    api_key="x",
    model_name="open-mixtral-8x22b",
    #model_path="./llm/Qwen2-0.5B-Instruct.Q8_0.gguf", #Qwen 2-0 5B Instruct Q6 GUFF
    # model_path="./llm/Meta-Llama-3-8B-Instruct.Q6_K.gguf", #Meta Llama 3 8B Instruct Q6 GUFF
    #model_path="./llm/TinyLlama-1.1B-Chat-v1.0-UltraQ-Imat-NEO1-Q8_0-imat.gguf", #TinyLlama 1 1.1B Chat Q6 GUFF
    temperature=0,
    max_tokens=512,
    top_p=1,
    # verbose=True,
    n_ctx=10000
)
#prompt = hub.pull("hwchase17/react")

ddgSearch = DuckDuckGoSearchAPIWrapper()
ddg_search_tool = DuckDuckGoSearchResults(verbose=True, api_wrapper=ddgSearch)

tools = [ddg_search_tool]

llm_with_tools = llm.bind_tools(tools=tools)

template = '''Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
TOOLS:
------
Assistant has access to the following tools:
{tools}
+-If the human ask for something that you do not now, use a tool to search for a information.
To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]. Ever use the 'Action Input + tool name' to call for a tool.
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
Begin!
New input: {input}
{agent_scratchpad}'''
print(template)
prompt = PromptTemplate.from_template(template)

question = "Show me the link of instagram from this profile: mateusfelipe.x"

# response = llm_with_tools.invoke(input=question)

agent = create_react_agent(llm=llm_with_tools, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
response = agent_executor.invoke({
    "input": f"{question}"
})
# response = agent_executor.invoke({
#     "input": f"{question}",
# })

print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")
print(response)
print("------------------------------------------------------------\n------------------------------------------------------------\n------------------------------------------------------------")