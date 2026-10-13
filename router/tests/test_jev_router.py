"""Unit tests for jev_router.py with a fake Jev (no network)."""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import jev_router as jr  # noqa: E402

POLICY_FILE = ROOT / "policy.json"
DOC = json.loads(POLICY_FILE.read_text(encoding="utf-8"))
K, DS, DS7, GF, G = "kimi-k3", "deepseek-v4.1-flash", "deepseek-v4-flash-0731", "glm-5.3-flash", "glm-5.3"
PRIMARY = {"math_proof": K, "math_answer": DS, "science_advanced": K, "science_basic": DS, "humanities_expert": K,
           "coding_direct": DS, "text_transformation": DS, "text_reasoning": K, "logic_puzzle": DS,
           "writing_creative": DS, "data_analysis_direct": DS, "general_direct": DS,
           "coding_agent": K, "research_agent": GF, "workflow_agent": DS, "service_agent": G}
LONG = {**{c: K for c in PRIMARY}, "data_analysis_direct": DS7}
CLASSES = tuple(PRIMARY)
AGENT = ("coding_agent", "research_agent", "workflow_agent", "service_agent")
DIRECT = tuple(c for c in CLASSES if c not in AGENT)


class FakeJev:
    """Answers with probabilities peaked on the class named 'CLS=<class>' in the request, or a scripted answer."""

    def __init__(self, status=200, answer=None, raw=None):
        self.status, self.answer, self.raw = status, answer, raw
        self.calls, self.bodies, self.headers = 0, [], []

    def handler(self, request: httpx.Request):
        self.calls += 1
        body = json.loads(request.content)
        self.bodies.append(body)
        self.headers.append(dict(request.headers))
        if self.raw is not None:
            return httpx.Response(self.status, content=self.raw)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": "unavailable"})
        answer = self.answer
        if answer is None:
            cls = re.search(r"CLS=(\w+)", json.dumps(body["state"])).group(1)
            answer = {"choice": cls, "confidence": 0.85, "probabilities": {c: (0.85 if c == cls else 0.01) for c in CLASSES}}
        return httpx.Response(200, json={"answers": {"task_class": answer}})


def make_router(fake=None, *, fits=jr.always_fits, store=None, policy=None, api_key="test-key", **kw):
    policy = policy or jr.Policy.load(POLICY_FILE)
    fake = fake or FakeJev()
    jev = jr.JevClient.from_policy(policy, api_key=api_key, client=httpx.Client(transport=httpx.MockTransport(fake.handler)),
                                   backoff_s=0, **kw)
    return jr.Router(policy, jev, fits=fits, store=store), fake


def conv(cls, extra="", turns=1, system="You are a helpful assistant."):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"CLS={cls} {extra}"}]
    for i in range(turns - 1):
        msgs += [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": f"next {i}"}]
    return msgs


def reply(content="the answer", finish="stop", **message):
    return {"choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": content, **message}}]}


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---- policy ----------------------------------------------------------------------------------------------------

def test_policy_table_and_prompt():
    p = jr.Policy.load(POLICY_FILE)
    assert p.class_names == CLASSES and p.jev_failure_model == K
    assert {c: (v["kind"], v["primary"], v["long_context"]) for c, v in p.classes.items()} == \
        {c: ("agent" if c in AGENT else "direct", PRIMARY[c], LONG[c]) for c in CLASSES}
    assert p.handoff == {"model": K, "kinds": ("direct",), "triggers": ("empty", "length", "error")}
    raw = p.prompt_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == p.prompt_sha256
    q = jr.load_questions(p.prompt_path, p.prompt_sha256, p.prompt_dimension)
    doc = json.loads(raw)
    assert list(q) == ["task_class"] and list(q["task_class"]["criteria"]) == list(CLASSES)
    assert q["task_class"]["type"] == "choice"
    assert q["task_class"]["instructions"] == doc["preamble"] + "\n\n" + doc["questions"]["task_class"]["instructions"]


def test_chains():
    p = jr.Policy.load(POLICY_FILE)
    assert p.chain("data_analysis_direct") == [DS, DS7, K]
    assert p.chain("math_answer") == [DS, K] and p.chain("math_proof") == [K] and p.chain("service_agent") == [G, K]
    assert p.chain(None) == [K]


def test_prompt_sha_mismatch_is_refused(tmp_path):
    doc = copy.deepcopy(DOC)
    (tmp_path / "jev_prompt.json").write_bytes((ROOT / "jev_prompt.json").read_bytes() + b" ")
    (tmp_path / "policy.json").write_text(json.dumps(doc))
    p = jr.Policy.load(tmp_path / "policy.json")
    with pytest.raises(ValueError, match="sha256"):
        jr.load_questions(p.prompt_path, p.prompt_sha256, p.prompt_dimension)


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(long_context_next={}), "not on the chain of data_analysis_direct"),
    (lambda d: d["handoff"].update(triggers=["empty", "timeout"]), "unknown handoff"),
    (lambda d: d["handoff"].update(kinds=[]), "needs model"),
    (lambda d: d["classes"]["math_proof"].update(kind="tool"), "math_proof"),
])
def test_policy_is_validated(change, match):
    doc = copy.deepcopy(DOC)
    change(doc)
    with pytest.raises(ValueError, match=match):
        jr.Policy(doc, base_dir=ROOT)


# ---- class -> model --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", CLASSES)
def test_each_class_goes_to_its_primary_model(cls):
    router, fake = make_router()
    d = router.route(conv(cls))
    assert (d.model, d.cls, d.kind) == (PRIMARY[cls], cls, "agent" if cls in AGENT else "direct")
    assert d.new_session and not d.jev_failure and not d.moved and fake.calls == 1


def test_jev_request_body():
    router, fake = make_router()
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather. Second sentence.",
                                               "parameters": {"type": "object"}}}]
    msgs = conv("general_direct")
    router.route(msgs, tools)
    body = fake.bodies[0]
    assert body["model"] == "jev-latest" and body["state"] == jr.build_view(msgs, tools)
    assert body["questions"] == router.jev.questions and fake.headers[0]["authorization"] == "Bearer test-key"


def test_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "from-env")
    router, fake = make_router(api_key=None)
    router.route(conv("math_answer"))
    assert fake.headers[0]["authorization"] == "Bearer from-env"


# ---- class selection -------------------------------------------------------------------------------------------

def test_argmax_probability_wins_over_choice():
    probs = {c: 0.0 for c in CLASSES} | {"general_direct": 0.4, "math_proof": 0.6}
    router, _ = make_router(FakeJev(answer={"choice": "general_direct", "probabilities": probs}))
    d = router.route(conv("x"))
    assert (d.cls, d.model) == ("math_proof", K)


def test_select_class_rules():
    assert jr.select_class({"choice": "logic_puzzle"}) == "logic_puzzle"
    assert jr.select_class({"choice": "logic_puzzle", "probabilities": {}}) == "logic_puzzle"
    assert jr.select_class({"choice": "a", "probabilities": {"b": 0.5, "c": 0.5}}) == "b"         # tie: listed first
    assert jr.select_class({"choice": "a", "probabilities": {"b": "high", "c": 0.1}}) == "c"      # non-numbers ignored


# ---- Jev failure -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fake", [
    FakeJev(status=500), FakeJev(status=429), FakeJev(raw=b"not json"),
    FakeJev(answer={"choice": "translation_agent", "confidence": 0.9}),
    FakeJev(answer={"choice": "nope", "probabilities": {"nope": 0.9}}),
])
def test_jev_failure_goes_to_failure_model_and_sticks(fake):
    router, _ = make_router(fake, attempts=2)
    d = router.route(conv("general_direct"))
    assert (d.model, d.cls, d.kind) == (K, None, None) and d.jev_failure and fake.calls == 2   # tried again first
    d2 = router.route(conv("general_direct", turns=2))
    assert d2.model == K and d2.jev_failure and not d2.new_session and fake.calls == 2


def test_jev_network_error_and_missing_key():
    def down(request):
        raise httpx.ConnectError("down")
    policy = jr.Policy.load(POLICY_FILE)
    jev = jr.JevClient.from_policy(policy, api_key="k", client=httpx.Client(transport=httpx.MockTransport(down)), backoff_s=0)
    assert jr.Router(policy, jev).route(conv("math_answer")).model == K
    router, fake = make_router(api_key="")
    assert router.route(conv("math_answer")).model == K and fake.calls == 0


# ---- sessions --------------------------------------------------------------------------------------------------

def test_one_jev_call_per_session():
    router, fake = make_router()
    for n in (1, 2, 3):
        d = router.route(conv("service_agent", turns=n))
        assert d.model == G and d.new_session is (n == 1)
    assert fake.calls == 1


def test_session_key_rule():
    msgs = [{"role": "developer", "content": [{"type": "text", "text": "be "}, {"type": "text", "text": "brief"}]},
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    expected = hashlib.sha256("be brief".encode() + b"\x00" + b"hi").hexdigest()[:32]
    assert jr.session_key(msgs) == expected == jr.session_key(msgs[:2])
    no_system = [{"role": "user", "content": "hi"}]
    assert jr.session_key(no_system) == hashlib.sha256(b"\x00hi").hexdigest()[:32]
    blank_first = [{"role": "system", "content": ""}, {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    assert jr.session_key(blank_first) == hashlib.sha256(b"be brief\x00hi").hexdigest()[:32]
    assert jr.session_key([{"role": "system", "content": "s"}]) is None
    assert jr.session_key(msgs, session_id="task-7") == "task-7"


def test_requests_without_a_user_message_have_no_session():
    router, fake = make_router(FakeJev(answer={"choice": "general_direct"}))
    for _ in range(2):
        d = router.route([{"role": "system", "content": "s"}])
        assert d.session_key is None and d.new_session
    assert fake.calls == 2


def test_client_session_id_takes_precedence():
    router, fake = make_router()
    a = router.route(conv("service_agent"), session_id="task-1")
    b = router.route(conv("service_agent"), session_id="task-2")
    c = router.route(conv("service_agent"))
    assert a.session is not b.session and fake.calls == 3 and c.session_key == jr.session_key(conv("service_agent"))


def test_identical_openings_share_a_session():
    router, fake = make_router()
    router.route(conv("logic_puzzle"))
    d = router.route(conv("logic_puzzle"))
    assert not d.new_session and fake.calls == 1


def test_sliding_expiry():
    clock = Clock()
    router, fake = make_router(store=jr.SessionStore(idle_expiry_s=100, clock=clock))
    router.route(conv("math_answer"))
    for _ in range(3):                         # each request keeps the session alive
        clock.t += 90
        assert not router.route(conv("math_answer", turns=2)).new_session
    clock.t += 101
    assert router.route(conv("math_answer", turns=2)).new_session and fake.calls == 2


def test_store_drops_least_recently_used():
    store = jr.SessionStore(max_sessions=2)
    for k in "abc":
        store.put(k, jr.Session("general_direct", "direct", [DS, K], DS))
    assert store.get("a") is None and store.get("b") and store.get("c")


# ---- handoff ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("content", ["", "  \n\t ", None, [{"type": "text", "text": " "}]])
def test_handoff_on_blank_content(content):
    assert jr.needs_handoff(reply(content)) == "empty"


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_handoff_on_reasoning_only_reply(field):
    assert jr.needs_handoff(reply("", **{field: "long thinking that never states the answer"})) == "empty"


@pytest.mark.parametrize("content", ["a derivation that stops half way", ""])
def test_handoff_on_length(content):
    assert jr.needs_handoff(reply(content, finish="length")) == "length"


def test_handoff_on_error():
    assert jr.needs_handoff(None, error=True) == "error"
    assert jr.needs_handoff({"choices": []}) == "error"
    assert jr.needs_handoff(reply("", finish="error")) == "error"


def test_answers_do_not_trigger():
    call = {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
    assert jr.needs_handoff(reply("Paris")) is None
    assert jr.needs_handoff(reply(None, finish="tool_calls", tool_calls=[call])) is None
    assert jr.needs_handoff(reply([{"type": "text", "text": "42"}])) is None


@pytest.mark.parametrize("cls", [c for c in DIRECT if PRIMARY[c] == DS])
def test_direct_class_hands_off_to_kimi_and_session_stays(cls):
    router, fake = make_router()
    d = router.route(conv(cls))
    assert router.handoff(d, reply("")) == K
    later = router.route(conv(cls, turns=2))
    assert later.model == K and not later.new_session and fake.calls == 1
    assert later.session.moves == [(DS, K, "handoff_empty")]
    assert router.handoff(later, reply("", finish="length")) is None      # Kimi K3 serves: no further handoff


@pytest.mark.parametrize("trigger", [{"reply": reply("")}, {"reply": reply("x", finish="length")}, {"error": True}])
@pytest.mark.parametrize("cls", AGENT)
def test_no_handoff_for_agent_classes(cls, trigger):
    router, _ = make_router()
    d = router.route(conv(cls))
    assert router.handoff(d, **trigger) is None and router.route(conv(cls, turns=2)).model == PRIMARY[cls]


@pytest.mark.parametrize("cls", [c for c in DIRECT if PRIMARY[c] == K])
def test_no_handoff_when_kimi_serves(cls):
    router, _ = make_router()
    d = router.route(conv(cls))
    assert d.model == K and router.handoff(d, reply("", finish="length")) is None


def test_no_handoff_for_streamed_requests():
    router, _ = make_router()
    d = router.route(conv("general_direct"))
    assert router.handoff(d, reply(""), stream=True) is None and d.session.model == DS


def test_no_handoff_after_jev_failure():
    router, _ = make_router(FakeJev(status=500), attempts=1)
    d = router.route(conv("general_direct"))
    assert d.model == K and router.handoff(d, reply("")) is None


def test_handoff_from_long_context_model():
    big = lambda model, request: model != DS or "rows" not in json.dumps(request["messages"])    # noqa: E731
    router, _ = make_router(fits=big)
    d = router.route(conv("data_analysis_direct", extra="rows"))
    assert d.model == DS7 and d.moved and router.handoff(d, reply("partial", finish="length")) == K
    assert d.session.moves == [(DS, DS7, "input_size"), (DS7, K, "handoff_length")]


def test_no_handoff_when_kimi_cannot_hold_the_request():
    router, _ = make_router(fits=lambda model, request: model != K)
    d = router.route(conv("general_direct"))
    assert d.model == DS and router.handoff(d, reply("")) is None and d.session.model == DS


def test_handoff_target_is_pure():
    p = jr.Policy.load(POLICY_FILE)
    s = jr.Session("math_answer", "direct", p.chain("math_answer"), DS)
    assert jr.handoff_target(p, s, DS, reply("")) == K and s.model == DS
    assert jr.handoff_target(p, s, DS, reply("ok")) is None
    assert jr.handoff_target(p, s, K, reply("")) is None
    assert jr.handoff_target(p, None, DS, reply("")) is None


# ---- long-context chain ----------------------------------------------------------------------------------------

def sized(limits):
    """fits by message character count, for tests only."""
    return lambda model, request: sum(len(str(m.get("content"))) for m in request["messages"]) <= limits[model]


def test_first_request_that_outgrows_the_primary_model():
    router, _ = make_router(fits=sized({DS: 100, DS7: 1000, K: 10_000, G: 100, GF: 100}))
    d = router.route(conv("math_answer", extra="x" * 500))
    assert d.model == K and d.moved and d.session.moves == [(DS, K, "input_size")]


def test_chain_moves_forward_only():
    router, fake = make_router(fits=sized({DS: 100, DS7: 1000, K: 10_000, G: 100, GF: 100}))
    base = conv("data_analysis_direct")
    assert router.route(base).model == DS
    m2 = base + [{"role": "user", "content": "r" * 500}]
    d2 = router.route(m2)
    assert d2.model == DS7 and d2.moved and not d2.new_session
    m3 = m2 + [{"role": "user", "content": "r" * 2000}]
    assert router.route(m3).model == K
    assert router.route(base).model == K and fake.calls == 1                 # never moves back
    assert [m[1] for m in d2.session.moves] == [DS7, K]


def test_nothing_fits():
    router, _ = make_router(fits=lambda model, request: False)
    d = router.route(conv("general_direct"))
    assert d.model is None and d.session.model == K
    assert router.route(conv("general_direct", turns=2)).model is None


def test_provider_rejects_input_as_too_long():
    router, _ = make_router()
    d = router.route(conv("service_agent"))
    assert router.on_input_too_long(d) == K and router.route(conv("service_agent", turns=2)).model == K


# ---- view ------------------------------------------------------------------------------------------------------

def test_view_flattens_messages():
    msgs = [{"role": "system", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "u"}},
                                         {"type": "text", "text": "here"}, {"type": "input_audio"}, None]},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": 42},
            "not a message"]
    original = copy.deepcopy(msgs)
    assert jr.build_view(msgs) == {"messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "look\nhere", "non_text_parts": 2},
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "42"}]}
    assert msgs == original


def test_view_compact_tools():
    tools = [{"type": "function", "function": {"name": "send_email", "description": "Send an email. Use it for X.\nMore.",
                                               "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}}},
             {"name": "run_query", "description": "Run SQL! Returns rows."},
             {"type": "function", "function": {"description": "nameless"}},
             "lookup_order", None,
             {"type": "function", "function": {"name": "summarise", "description": "y" * 300}}]
    original = copy.deepcopy(tools)
    view = jr.build_view([{"role": "user", "content": "hi"}], tools)
    assert view["tools"] == [
        {"name": "send_email", "category": "messaging", "description": "Send an email."},
        {"name": "run_query", "category": "database", "description": "Run SQL!"},
        {"name": "lookup_order", "category": "lookup_operations", "description": ""},
        {"name": "summarise", "category": "summarise_operations", "description": "y" * 160}]
    assert "tool_categories" not in view and tools == original
    assert "tools" not in jr.build_view([{"role": "user", "content": "hi"}], [])


def test_long_tool_list_becomes_category_counts():
    tools = [{"type": "function", "function": {"name": f"crm_update_{i}", "description": "Update a CRM record " * 5}}
             for i in range(60)] + [{"type": "function", "function": {"name": "send_slack", "description": "Post."}}]
    assert jr.estimate_tokens(jr.compact_tools(tools)) > jr.TOOL_LIST_BUDGET
    view = jr.build_view([{"role": "user", "content": "hi"}], tools)
    assert view["tool_categories"] == {"crm_operations": 60, "messaging": 1} and "tools" not in view


def test_custom_view_hook():
    seen = []

    def short_view(messages, tools):
        seen.append(len(messages))
        return {"messages": [{"role": "user", "content": "CLS=coding_agent (condensed)"}]}
    policy = jr.Policy.load(POLICY_FILE)
    fake = FakeJev()
    jev = jr.JevClient.from_policy(policy, api_key="k", client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    d = jr.Router(policy, jev, build_view=short_view).route(conv("general_direct"))
    assert d.cls == "coding_agent" and seen == [2] and fake.bodies[0]["state"]["messages"][0]["content"].endswith("(condensed)")


def test_estimate_tokens():
    assert jr.estimate_tokens("abcde") == 2 and jr.estimate_tokens("12345") == 5
    assert jr.estimate_tokens("\u4e2d\u6587") == 2 and jr.estimate_tokens({"a": 1}) == round(1 + 7 / 2.5)
