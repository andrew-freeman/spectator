from app.response.response_builder import build_response


def _base_kwargs():
    return dict(
        reflection_output={},
        actor_output={},
        critic_output={},
        governor_decision={},
        tool_results=[],
    )


def test_chat_mode_returns_natural_language():
    response = build_response(
        user_message="hello there",
        reflection_output={"intent": "chat", "context": {"chat_mode": True}},
        actor_output={},
        critic_output={},
        governor_decision={},
        tool_results=[],
        identity_profile={"name": "Spectator", "role": "assistant", "environment": "lab"},
    )
    assert "Spectator" in response
    assert "hello there" in response


def test_query_mode_formats_tool_results():
    response = build_response(
        user_message="what are the temps?",
        reflection_output={"intent": "query", "context": {"query_mode": True}},
        actor_output={},
        critic_output={},
        governor_decision={"verdict": "query_mode"},
        tool_results=[
            {"tool": "read_gpu_temps", "status": "ok", "result": {"gpu_temps": [55, 57]}}
        ],
        identity_profile={"name": "Spectator"},
    )
    assert "Here is what I found" in response
    assert "55" in response


def test_identity_question_uses_description():
    response = build_response(
        user_message="Who are you?",
        **_base_kwargs(),
        identity_profile={"description": "Custom identity"},
    )
    assert response == "Custom identity"


def test_action_confirmation_mentions_fan_speed():
    response = build_response(
        user_message="set the fan",
        reflection_output={"intent": "command", "context": {"command_mode": True}},
        actor_output={},
        critic_output={},
        governor_decision={"verdict": "trust_actor"},
        tool_results=[
            {"tool": "set_fan_speed", "status": "ok", "result": {"fan_speed": 45, "reason": "keep gpu cool"}}
        ],
        identity_profile={"name": "Spectator"},
    )
    assert "Fan speed set to 45%" in response
