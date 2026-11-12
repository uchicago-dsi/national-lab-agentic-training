# Activities for Day 2

In Day 2, we will move from bespoke agentic workflows to coding agents. 
To demystify how coding agents work, we will first build a simple coding agent.
Then we will move to using commercial agents

For model access, use the OpenAI compatible API to the ALCF inference endpoints.
Full documentation can be found [here](https://docs.alcf.anl.gov/services/inference-endpoints/#2-authenticate).
To authenticate and set your api token, run these commands:

```
$ uvx alcf-tokens login inference
$ export OPENAI_BASE_URL=https://inference-api.alcf.anl.gov/resource_server/minerva/api/v1
$ export OPENAI_API_KEY=$(uvx alcf-tokens get-token inference)
```

Tokens are valid for 48 hours, so re-run the last command if you start getting authentication errors.

All of the scripts default to the `beam-1.0-bf16` model on Minerva. Use `-m/--model` to try a different model.
The models available on each cluster are listed [here](https://docs.alcf.anl.gov/services/inference-endpoints/models/).
If you switch to a model on a different cluster (e.g. Sophia, Metis, or Tara), you also need to change `OPENAI_BASE_URL` to that cluster's endpoint.

All of the commands below should be run from this directory (`activities/day_2`).

## Activity 01: Building a Coding Agent.

This activity will use the mechanisms from day 1 to build a coding agent.
This activity is inspired by this [guide](https://www.mihaileric.com/The-Emperor-Has-No-Clothes/), but slightly modified for an OpenAI compatible API and modern, server-side tool parsing.
The coding agent uses the same native tool calling from the previous class.
However, instead of implementing a bespoke tool, we will give the agent access to the commandline.

You must write the `run_command` tool to allow the agent to run generic shell commands, and the tool calling loop inside the coding agent loop.
We purposely use commands that were close to the raw model calls to understand how the internals of a coding agent work.
After you are done, ask it to do two different tasks:

(1) Ask it to write a `hello_world.py` program to test if all the functionality works

(2) Find a paper on arxiv, produce multiple reviews of the paper, and then a unified score.

### Extensions:

Cost estimates: Track the number of tokens used by your coding agent and produce a cost estimate for the running conversation. Use Together.ai pricing to determine the cost per token of open source models.

Tool Limits: Reading a large file or executing a command can overwhelm the agent context very quickly. Change the read tool to limit the size of the results to reduce context. How does your approach limit the cablities of the tool?

Safe Command Classification: Currently the code asks for your approval before any tool call. This ensures that the agent cannot run destructive commands. Implement a basic classifier that allows some safe commands and only asks on commands that may potentially be dangerous. What commands do you consider dangerous and how do you decide? Are certain calls always safe?

## Activity 02: Create a Skill

Now that we have an understanding of how coding agents work, we will move to using a commercial coding agent. In these activities, we will specifically use Pi because it is light-weight, highly compatible with open source models, and customizable. (Harnesses like Claude Code are tuned to work well with anthropic models specifically, and other harnesses like OpenCode or Hermes have a longer context-overhead that can cause shorter-context open source models to perform poorly.)

### Setup Pi with the ALCF Inference Service
Follow these [instructions](https://docs.alcf.anl.gov/services/inference-endpoints/agents/#automatic-configuration-w-alcf-ai) to set up Pi to use the ALCF inference service.

These two commands should be sufficient. The first installs Pi, the second configures it to use the inference service. 

```
$ curl -fsSL https://pi.dev/install.sh | sh
$ alcf-ai agent configure pi
```
## Write a Skill

Add a skill to the coding agent to perform a computation that's relevant to your domain. If you cannot come up with anything, write a skill for using Globus Compute that we discussed last time.

In the project root directory create a folder `.agents/skills/`.

The skill that you create should have this format:
```
pdf-tools/
├── SKILL.md
├── scripts/
│   └── extract.sh
├── references/
│   └── formats.md
└── assets/
    └── template.json
```
For reference on how to write Pi skills look at the [documentation](https://pi.dev/docs/latest/skills).

Compare how token use differs performing a task with the skill and without the skill.

## Activity 03: Context-Rot Evaluation

Now that we are using a coding agent, we can see that we can accomplish much more in a single activity. Please use Pi or your favorite coding agent to complete these tasks.

First, find a recent conference or journal from your field. We want them to be recent papers so that the model cannot use its parametric data. Download 20-50 papers from the proceedings into a new `sample_papers/` folder. (The prompt that I used for this is in `03_prompts.txt`, if you cannot access papers/journals from your domain, you can use this to follow along.)

Now, extract questions from each of the papers. We want questions that can be answered both with just lexical similarity and that require semantic understanding. Ask your coding agent to extract questions to a json file, with 4 fields: 'question', 'answer', 'category', and a 'source_paper' field.
Spot check some of the questions to validate that the answers are reasonable and contained in the paper.

We want to perform two different evaluations. First, there is a benchmarking script that queries a language model and tests its ability to recall information from different places in your context. Look at the parameters of the `03_long_context_benchmark.py` script to examine what you can change and investigate. You can run the script with:

```
uv run python 03_long_context_benchmark.py --papers sample_papers --questions questions.json -n 10
```

We also want to see how a coding agent navigates these papers when it has the ability to use tools (like find and grep). Start a new Pi session inside the `sample_papers/` folder. Outside of that folder Pi will have access to the questions and the answers, and will likely use them for the task.
Ask questions that you extracted earlier and examine how the agent tackles the question.
How did this differ from the approach taken by the benchmark?

 ## Activity 04: RAG

 ### Part 1: Vector Database Tool

Use Chroma to add a RAG retrieval tool to the paper question and answering benchmark.
The starter code is provided in `04_long_context_with_rag.py`.
Notice that we moved to using the OpenAI Agents SDK instead of our own agentic loop. 
The Agents SDK implements the same pattern that you implemented in your coding agent but we presume it is more robust.

First, chunk the papers, and use an embedding model on the ALCF inference service to create a vector embedding for each chunk.
Store each chunk into chroma.
Then write a tool that allows the model to query the vector database with plain text. 
The plain text should be embedded using the same embedding model that the papers were ingested with.
Run the script with:

```
uv run python 04_long_context_with_rag.py --papers sample_papers --questions questions.json -n 20
```
 

 ### Part 2: Connect the vector DB to the Agent

We will now connect RAG with Pi to examine how a coding agent can use RAG in a similar context.

Add the MCP server to Pi. Create `.pi/mcp.json` inside of the project root directory. Write a block:
```
{
  "mcpServers": {
    "chroma": {
      "command": "uvx",
      "args": [
        "chroma-mcp",
        "--client-type",
        "persistent",
        "--data-dir",
        "</path/to/your/data/directory>"
      ]
    }
  }
}
```


Make the data directory point to the absolute path to a `chroma` folder in this directory.

Then ask your coding model to "write a python script to ingest the collected papers into the chroma database using the default embedding model" and run the script.

Check to see if the MCP server is connected to pi:
```
pi mcp list
```

Then run pi and ask about the ingested papers.