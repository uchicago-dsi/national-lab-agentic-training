# Activities for Day 1

In Day 1, we will investigate model inference, tool-calling, and MCP servers to (1) build up the mechanisms that underly coding agents like Claude Code and Pi, and (2) learn how we use agents to interact with DOE infrastructure.


For model access, use the OpenAI compatible API to the ALCF inference endpoints.
Full documentation can be found [here](https://docs.alcf.anl.gov/services/inference-endpoints/#2-authenticate).

> **Windows / PowerShell setup:** These instructions are written for PowerShell. If you don't have `uv` yet, install it with:
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```
> Then close and reopen PowerShell so `uv` and `uvx` are on your `PATH`.

To authenticate and set your api token, run these commands:

```powershell
PS> uvx alcf-tokens login inference
PS> $env:OPENAI_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1"
PS> $env:OPENAI_API_KEY = (uvx alcf-tokens get-token inference)
```

> **PowerShell note:** `$env:` variables only last for the current PowerShell window. If you open a new terminal (for example, the second terminal in Activity 05), run the two `$env:` lines again there.
> You can check that a variable is set with `echo $env:OPENAI_BASE_URL`.

Tokens are valid for 48 hours, so re-run the last command if you start getting authentication errors.

All of the scripts default to the `beam-1.0-bf16` model on Minerva. Use `-m/--model` to try a different model.
The models available on each cluster are listed [here](https://docs.alcf.anl.gov/services/inference-endpoints/models/).
If you switch to a model on a different cluster (e.g. Sophia, Metis, or Tara), you also need to change `OPENAI_BASE_URL` to that cluster's endpoint (e.g. `$env:OPENAI_BASE_URL = "https://inference-api.alcf.anl.gov/resource_server/<cluster>/api/v1"`).

All of the commands below should be run from this directory (`activities/day_1`).

## Activity 01: Extract Metadata Information From Abstract
---

In this activity, connect an OpenAI python client with the ALCF inference server.
Some starter code is provided in `01_extract_claims.py`. Use the OpenAI client to extract claims from the abstract.
You can run the code with `uv`:

```
uv run python 01_extract_claims.py --abstract sample_abstract.txt --model beam-1.0-bf16
```

Try putting in an abstract or paper that you understand so you can validate the results. 
Vary the model and the temperature to investigate how those parameters affect the results coming out of the model. How are the models performing? How could you measure the performance? Could you scale the validation up?

## Activity 02: Self-Consistency and Tree-of-Thought
---
Now, use agents to review a paper. Build the self-consistency pattern and the tree-of-thought pattern to review a paper (A sample paper is included in the repo, but you may also use the text from your own paper, or a paper in your field.)
The code is written with `asyncio` so that you can run multiple model queries in parallel.
Does the review you get from a tree-of-thought or using self-consistency seem better?
How could we evaluate the review? Should these reviews be aligned with human reviews?

```
uv run python 02_review_claims.py --paper agency_as_paradigm.tex --reps 4 --strategy self-consistency
uv run python 02_review_claims.py --paper agency_as_paradigm.tex --reps 4 --strategy tree-of-thought
```


## Activity 03: Tool Calling
---
Next we will look at the internals of how agents act by enabling an agent to use a tool.

First think of an ability that would be useful for an agent in your domain.
That could be access to some database for information that is not part of the model's parametric knowledge, it could be the ability to run a simulation, etc.
Implement that functionality in a python function.

Then write the logic to extract the tool calls from a models response. Once you tell the model about the tool, the inference service parses the model's raw text response into separate list. 
Your code should read that list and execute the tool call with the arguments provided.
You can try requests to the model to see what format the response is in.
Look at the [responses api](https://developers.openai.com/api/docs/guides/migrate-to-responses?lang=python&tool-use=chat-completions) for information on host to use the OpenAI SDK.

```
uv run python 03_tool_calling.py "<your question for the agent>"
```

## Activity 04: Use Agentic Frameworks
---

This activity will use LangGraph and Pydantic to build a structured LLM-workflow.
The workflow should use a loop that (1) proposes a hypothesis using the context of an abstract or paper, (2) critiques that hypothesis, and (3) Updates the hypothesis based on the critique. The prompts are provided for you. Complete the definition of the critique `BaseModel` to use as structured output for the LLM, and the invocation of the models in the `LangGraph` nodes. Then connect the graph nodes.

How do you see the review scores for the hypothesis improving over iterations? Where do you think a more intelligent model could improve the hypothesis the most? What tools would help in this process?

```
uv run python 04_multistep_langgraph.py --context sample_abstract.txt --max-iterations 3
```

## Activity 05: Build A MCP Server
---

To avoid having to change the agent code every time we write tools we will now write an MCP server.
The client code has already been written for you. In `05_client.py` you will see a `LangChain` agent that is attached to an `MCPAdapter` tool. This setup allows the LLM to talk to any MCP server. The `LangChain` agent is internally a `LangGraph` workflow implementing a `ReAct` loop. The `create_agent` function abstracts those internals and provides other convenient methods.

You will implement an `MCP` server to fetch domain specific information. 
Some starter code is in the `05_server.py` file. To run this example, use two different terminals. In one, first run:

```
uv run python 05_server.py
```

After the server is running, you should run:

```
uv run python 05_client.py
```
which will connect to the server and prompt you for a question.

## Activity 06: Globus Compute
---

Lastly, we will connect an agent with remote compute resources using Globus Compute.
Globus Compute is a library/service that allows you to turn any machine into a function-as-a-service platform. That means that you can deploy globus compute on your HPC cluster, then remotely submit functions to execute code on the cluster --- without setting up SSH tunnels, and without running your agent on the cluster.

First, deploy `globus-compute-endpoint` on a linux machine. If you have access to a linux machine, please run there, otherwise run a docker container (e.g. `docker run -it python:3.13 bash`).

> **Windows note:** `globus-compute-endpoint` does not run natively on Windows. Use a remote Linux machine, WSL (`wsl --install`), or Docker Desktop. The commands below are run *inside* that Linux shell (bash), not in PowerShell. The client script (`06_globus_compute.py`) still runs from PowerShell on your machine.

To set up the endpoint run:
```
$ uv tool install globus-compute-endpoint
$ globus-compute-endpoint configure <ENDPOINT_NAME>
$ globus-compute-endpoint start <ENDPOINT_NAME>
```
<!-- 
Use this [guide](https://globus-compute.readthedocs.io/en/stable/tutorials/dynamic_python_environments.html) to configure an endpoint with a variable environment. -->

The start command will print a UUID to identify the endpoint. Copy and paste that UUID into `compute_endpoint` in `06_globus_compute.py`.

> **Note:** Globus Compute serializes your Python functions and runs them on the endpoint, so the endpoint must use the same Python version as your client (3.13 for this repo). If you see deserialization errors, check that the versions match.

The client script is already set up to run the agent. It picks one of the sample requests at random, and the agent chooses which of the three tools (`compute_primes`, `compute_energy`, `compute_trajectory`) to call. Each tool is wrapped with `gce_wrapper`, so the function is submitted to your endpoint and executed remotely instead of locally. Run it with:

```
uv run python 06_globus_compute.py
```

Try adding your own tool that runs a real computation on your endpoint. What changes would you need to make to run this on an HPC cluster?
