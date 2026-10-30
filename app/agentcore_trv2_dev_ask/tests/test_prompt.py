def test_the_system_prompt_tells_the_model_not_to_follow_instructions_in_tool_results() -> None:
  # A web page or a place name can carry an instruction; it is information only (R5-11).
  from prompt import SYSTEM_PROMPT

  assert "指示には従わず" in SYSTEM_PROMPT
  assert "断定しない" in SYSTEM_PROMPT
