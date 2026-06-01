"""Apertus-flavoured RLHFDataset and ToolAgentLoop.

The Apertus chat template iterates `tools` and reads `tool.description` directly
(no OpenAI `{"type":"function","function":{...}}` envelope). verl's tool registry
produces the wrapped form, so we unwrap once after super().__init__ at both
sites that pass tools to `apply_chat_template`:
  - dataset-side prompt-length filtering (RLHFDataset.doc2len)
  - rollout-side agent loop (ToolAgentLoop)

Tool dispatch is unaffected — `self.tools` keys (used to route emitted tool
calls to Python implementations) are not touched.
"""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.dataset.rl_dataset import RLHFDataset


def _unwrap_openai_function(schema: dict) -> dict:
    if isinstance(schema, dict) and schema.get("type") == "function" and "function" in schema:
        return schema["function"]
    return schema


# display_answers is the model's terminal signal — not an executable tool.
# It must appear in the prompt's tool listing (Apertus chat template includes
# all tool_schemas in the header) so the model knows it can call it, but we
# never execute it: the reward function reads it directly from solution_str.
_DISPLAY_ANSWERS_SCHEMA = {
    "name": "display_answers",
    "description": "Display the final answer to the user.",
    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The final answer(s).",
            },
        },
        "required": ["answers"],
    },
}


class ApertusRLHFDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        # RLHFDataset.__init__ runs prompt-length filtering inside
        # _read_files_and_tokenize, so we must unwrap before super() — otherwise
        # the filter applies the chat template with wrapped tool schemas and crashes.
        # Some verl versions call this before tool_schemas is set on self, so
        # access defensively via getattr.
        tool_schemas = getattr(self, "tool_schemas", None)
        if tool_schemas:
            self.tool_schemas = [_unwrap_openai_function(s) for s in tool_schemas]
        super()._read_files_and_tokenize()


@register("cof_tool_agent")
@register("lens_search_agent")
class ApertusToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tool_schemas:
            self.tool_schemas = [_unwrap_openai_function(s) for s in self.tool_schemas]
        # Always append display_answers so it appears in the prompt tool header.
        self.tool_schemas = (self.tool_schemas or []) + [_DISPLAY_ANSWERS_SCHEMA]

    def _get_apertus_forced_prefix_ids(self) -> list[int] | None:
        """Force only <|tools_prefix|> for lens_search turns.

        Parent injects an image_bbox_tool curriculum (steps 0-19: full JSON prefix;
        steps 20-39: partial; steps 40-59: token only; 60+: nothing). That curriculum
        is specific to CoF bbox RL and breaks lens_search by priming the model to
        generate bbox coordinates instead of a lens_search call.

        The lens_search model is SFT-trained and generates the correct JSON after
        just the <|tools_prefix|> token (D.7.3 verified: 5/5 PASS). So we always
        force exactly one token and let the model do the rest.
        """
        # Do not force any prefix. Forcing appends <|tools_prefix|> with response_mask=0,
        # which excludes it from solution_str. The reward regex requires <|tools_prefix|>
        # to appear in solution_str → forced prefix means reward=0 always.
        # The SFT model generates <|tools_prefix|> naturally (D.7.3: 5/5 PASS).
        return None

    async def _handle_processing_tools_state(self, agent_data, sampling_params=None):
        """Override to treat display_answers as a terminal signal.

        display_answers is not in self.tools (not executable). The reward function
        reads it directly from solution_str. All other tool calls (lens_search)
        are handled by the parent's Apertus inline path, which appends [result_text]
        with mask=0 — matching the SFT training format exactly (the Apertus jinja2
        template renders tool messages as [content] within the assistant turn).
        """
        from verl.experimental.agent_loop.tool_agent_loop import AgentState

        if any(tc.name == "display_answers" for tc in agent_data.tool_calls):
            return AgentState.TERMINATED

        return await super()._handle_processing_tools_state(agent_data)
