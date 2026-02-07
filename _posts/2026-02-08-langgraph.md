---
layout: post
title: "LangGraph 맛보기 feat. tool 디버깅"
date: 2026-02-04
tags: [langgraph]
---


# LangGraph 2가지 사용법
- graph api: state, nodes and edges
- functional api: persistence, memory, human-in-the-loop, and streaming
- 대개 graph api 사용함. 
- functional apis는 이미 다른 코드가 있고, 거기에 langgraph 접붙일 때, 코드 수정 최소화를 위해 사용

---

- Graph api Example code

<details>

```
# Step 1: Define tools and model

from langchain.tools import tool
from langchain.chat_models import init_chat_model


model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)


# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# Step 2: Define state

from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

# Step 3: Define model node
from langchain.messages import SystemMessage


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


# Step 4: Define tool node

from langchain.messages import ToolMessage


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

# Step 5: Define logic to determine whether to end

from typing import Literal
from langgraph.graph import StateGraph, START, END


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

# Step 6: Build agent

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


from IPython.display import Image, display
# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```

</details>

---
# tool call 탐구

- ministral 모델 학습시 아마 아래와 같은 학습을 했을 것:
```
[INST] What's 2+2? [/INST]
[TOOL_CALLS]  # <- 아래 log 보면 이 token이 생성되는 것을 확인할 수 있음
 [{"name": "calculator", "arguments": {...}}]
```
- 위와 같은 tool use 학습하지 않은 모델은 agent 모델로 사용이 불가능함
- tool call 할 때 실제 message body
- lm studio 사용해서 디버깅해봄

```
2026-02-08 01:13:16 [DEBUG]
 Received request: POST to /v1/chat/completions with body  {
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "1*2",
      "role": "user"
    }
  ],
  "model": "mistralai/ministral-3-14b-reasoning",
  "stream": false,
  "temperature": 0,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Adds `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "description": "Multiply `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "divide",
        "description": "Divide `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    }
  ]
}

# 모델의 tool call
2026-02-08 01:13:17 [DEBUG]
 [SamplingSwitch] Entering switch 'mistralV13ToolsSamplingSwitch' (triggered by string: "[TOOL_CALLS]") # <----------- tool call 토큰 생성함. tool use 학습의 산물.
2026-02-08 01:13:17  [INFO]
 [mistralai/ministral-3-14b-reasoning] Start to generate a tool call...
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Tool name generated:  multiply
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Accumulating tool call arguments:  {"
...
 [mistralai/ministral-3-14b-reasoning] Accumulating tool call arguments:  {"a": 1, "b": 2}
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Model generated a tool call.
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Model generated tool calls:  [multiply(a=1, b=2)]

# 모델 response
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Generated prediction:  {
  "id": "chatcmpl-37bu06198lu6b9atpcug9i",
  "object": "chat.completion",
  "created": 1770480796,
  "model": "mistralai/ministral-3-14b-reasoning",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [ #<-------------- tool_call format에 맞춰서 response
          {
            "type": "function",
            "id": "570498487",
            "function": {
              "name": "multiply",
              "arguments": "{\"a\":1,\"b\":2}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 285,
    "completion_tokens": 17,
    "total_tokens": 302
  },
  "stats": {},
  "system_fingerprint": "mistralai/ministral-3-14b-reasoning"
}

# langgraph multiply tool의 결과값을 다시 llm으로 post
# function과 그 결과값을 돌려줌
2026-02-08 01:13:18 [DEBUG]
 Received request: POST to /v1/chat/completions with body  {
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "1*2",
      "role": "user"
    },
    {
      "content": null,
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "id": "570498487",
          "function": {
            "name": "multiply",
            "arguments": "{\"a\": 1, \"b\": 2}"
          }
        }
      ]
    },
    {
      "content": "2",
      "role": "tool",
      "tool_call_id": "570498487"
    }
  ],
  "model": "mistralai/ministral-3-14b-reasoning",
  "stream": false,
  "temperature": 0,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "add",
        "description": "Adds `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "multiply",
        "description": "Multiply `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "divide",
        "description": "Divide `a` and `b`.\n\n    Args:\n        a: First int\n        b: Second int",
        "parameters": {
          "properties": {
            "a": {
              "type": "integer"
            },
            "b": {
              "type": "integer"
            }
          },
          "required": [
            "a",
            "b"
          ],
          "type": "object"
        }
      }
    }
  ]
}

# llm 모델의 최종 response
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Running chat completion on conversation with 4 messages.
2026-02-08 01:13:18  [INFO]
 [mistralai/ministral-3-14b-reasoning] Accumulated 1 tokens  The
...
2026-02-08 01:13:19  [INFO]
 [mistralai/ministral-3-14b-reasoning] Accumulated 14 tokens  The result of \(1 \times 2\) is **2**.
2026-02-08 01:13:19  [INFO]
 [mistralai/ministral-3-14b-reasoning] Model generated tool calls:  []
2026-02-08 01:13:19  [INFO]
 [mistralai/ministral-3-14b-reasoning] Generated prediction:  {
  "id": "chatcmpl-2ojj2bd99jrvma2y5bngh",
  "object": "chat.completion",
  "created": 1770480798,
  "model": "mistralai/ministral-3-14b-reasoning",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The result of \\(1 \\times 2\\) is **2**.",
        "tool_calls": []
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 305,
    "completion_tokens": 15,
    "total_tokens": 320
  },
  "stats": {},
  "system_fingerprint": "mistralai/ministral-3-14b-reasoning"
}
```

reference 
- https://docs.langchain.com/oss/python/langgraph/quickstart
- https://medium.com/ai-agents/langgraph-for-beginners-part-3-conditional-edges-16a3aaad9f31
- https://github.com/teddylee777/LangGraph-Advanced-Tutorial/tree/main
- https://langchain-ai.github.io/langgraph/tutorials/web-navigation/web_voyager/#agent-definition
- https://github.com/spmallick/learnopencv/blob/master/LangGraph-A-Visual-Automation-and-Summarization-Pipeline/browser_automation.py
- https://docs.mistral.ai/cookbooks/concept-deep-dive-tokenization-tool_calling