"""
Edge-case behavior for SupportAgent's LLM call path.

These lock in real, reproducible findings from live experiments against a
real Ollama server (llama3.2:latest) -- see the "Edge-case & context-limit
findings" section of PROJECT-FIX-PLAN.md for the full writeup with exact
prompts/responses. Summary of what was actually observed (not assumed):

- Empty context + empty question: SupportAgent applies no validation
  before calling the LLM. Unlike schema-lab's structured extraction, the
  model doesn't hallucinate a fabricated answer here -- with no schema to
  force conformance to, it just asks for clarification.
- Malformed/garbage input: same graceful "I don't understand, please
  clarify" behavior, no error.
- Oversized context (exceeding the explicit num_ctx=2048 set in
  OllamaClient's default_options): SupportAgent applies no client-side
  length check of its own. Ollama silently truncates the prompt from the
  *front* with no error or warning. A policy fact placed at the very start
  of a long context was lost; because the prompt template's "don't make up
  an answer" instruction sits close to the end (right before "Context:"),
  it survived truncation and the model correctly declined to answer rather
  than fabricating one -- a real, template-position-dependent safety net,
  not a guarantee.

These tests assert the *plumbing* behavior (no guard exists, whatever the
LLM returns is passed straight through) using the real recorded outputs as
mock return values, so a future accidental input-guard or truncation
change would be caught as an intentional behavior change, not silently.
"""

from unittest.mock import MagicMock, patch

from promptops_lab.agents.support_agent import SupportAgent


@patch("promptops_lab.agents.support_agent.get_llm_client")
def test_empty_context_and_question_reach_the_llm_unmodified(mock_get_llm_client):
    mock_client = MagicMock()
    # Real observed behavior: no schema to conform to, so the model asks
    # for clarification instead of hallucinating an answer.
    mock_client.generate.return_value = (
        "I'm ready to help. Go ahead and ask your question!"
    )
    mock_get_llm_client.return_value = mock_client

    agent = SupportAgent()
    response = agent.execute(version="v1", context="", question="")

    assert response == "I'm ready to help. Go ahead and ask your question!"
    prompt_sent = mock_client.generate.call_args.kwargs["prompt"]
    assert "Context:\n\n" in prompt_sent
    assert "Question: \n" in prompt_sent


@patch("promptops_lab.agents.support_agent.get_llm_client")
def test_malformed_input_reaches_the_llm_unmodified(mock_get_llm_client):
    mock_client = MagicMock()
    mock_client.generate.return_value = (
        "I don't see a question from the user. Could you please provide "
        "the question so I can assist you?"
    )
    mock_get_llm_client.return_value = mock_client

    agent = SupportAgent()
    garbage_context = "\x00\x01 garbage_garbage $%^&*()"
    response = agent.execute(version="v1", context=garbage_context, question="???!!!")

    assert "provide the question" in response
    prompt_sent = mock_client.generate.call_args.kwargs["prompt"]
    assert garbage_context in prompt_sent
    assert "???!!!" in prompt_sent


@patch("promptops_lab.agents.support_agent.get_llm_client")
def test_oversized_context_is_not_truncated_or_validated_client_side(
    mock_get_llm_client,
):
    mock_client = MagicMock()
    # Real observed behavior: the policy fact at the very start of the
    # oversized context was silently truncated away by Ollama; the "don't
    # make up an answer" instruction survived (it sits near the end of the
    # template, right before Context:), so the model declined honestly
    # instead of fabricating a refund-window number.
    mock_client.generate.return_value = (
        "I don't have that information. The text you provided appears to "
        "be a snippet of a policy or agreement, but it doesn't specify "
        "the refund window."
    )
    mock_get_llm_client.return_value = mock_client

    huge_context = (
        "IMPORTANT POLICY FACT: The refund window is exactly 45 days. "
        + ("Our platform supports many integrations. " * 5000)
    )
    agent = SupportAgent()
    response = agent.execute(
        version="v1",
        context=huge_context,
        question="How many days is the refund window according to the policy?",
    )

    # No client-side truncation happens in this lab's own code -- the full,
    # un-truncated context is what gets sent onward to the LLM.
    prompt_sent = mock_client.generate.call_args.kwargs["prompt"]
    assert huge_context in prompt_sent

    # The model declined rather than fabricating an answer -- but this is
    # the prompt template's instruction surviving truncation, not a
    # guarantee schema-lab-style structured extraction also gets.
    assert "don't have that information" in response
