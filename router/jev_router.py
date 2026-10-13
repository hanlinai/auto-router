"""Reference implementation of the Auto routing decision.

For each new conversation, Jev (a decision model from TypeSafe AI) picks one of the task classes listed in
policy.json; the policy maps the class to a model, and every later request of the conversation stays on that
model. This module holds the decision logic only:

* Policy: the class table, the long-context chain and the handoff rule (policy.json).
* build_view: what Jev is shown for the first request of a conversation.
* JevClient / select_class: the Jev call; the class is the one with the highest probability.
* session_key / SessionStore: how requests are grouped into sessions.
* Router.route: the model for one request.
* needs_handoff / handoff_target / Router.handoff: when a direct-answer request is sent again to the handoff model.

Forwarding requests to the models, retrying failed requests on the same model, and logging are left to the caller.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
KINDS = ("direct", "agent")
TRIGGERS = ("empty", "length", "error")


# ---- policy --------------------------------------------------------------------------------------------------

class Policy:
    """Class table: each class has a kind (direct | agent), a primary model and a long-context model."""

    def __init__(self, doc: dict, base_dir: str | Path = "."):
        self.version = doc["version"]
        self.classes = doc["classes"]
        self.class_names = tuple(self.classes)
        for name, c in self.classes.items():
            if c.get("kind") not in KINDS or not c.get("primary") or not c.get("long_context"):
                raise ValueError(f"class {name} needs a kind ({'|'.join(KINDS)}), a primary and a long_context model")
        self.long_context_next = dict(doc.get("long_context_next") or {})
        self.jev_failure_model = doc["jev_failure_model"]
        prompt = doc["jev_prompt"]
        self.prompt_path = Path(base_dir) / prompt["path"]
        self.prompt_sha256 = prompt["sha256"]
        self.prompt_dimension = prompt.get("dimension", "task_class")
        self.handoff = None
        h = doc.get("handoff")
        if h:
            if not (h.get("model") and set(h.get("kinds") or ()) and set(h.get("triggers") or ())):
                raise ValueError("handoff needs model, kinds and triggers")
            if not set(h["kinds"]) <= set(KINDS) or not set(h["triggers"]) <= set(TRIGGERS):
                raise ValueError(f"unknown handoff kind or trigger in {h}")
            self.handoff = {"model": h["model"], "kinds": tuple(h["kinds"]), "triggers": tuple(h["triggers"])}
            for name, c in self.classes.items():
                if c["kind"] in self.handoff["kinds"] and h["model"] not in self.chain(name):
                    raise ValueError(f"handoff model {h['model']} is not on the chain of {name}")

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), base_dir=path.parent)

    def kind(self, cls: str | None) -> str | None:
        return self.classes[cls]["kind"] if cls is not None else None

    def chain(self, cls: str | None) -> list[str]:
        """Models in the order a session may use them: primary, long-context, then long_context_next steps.
        Without a class (Jev failed) the chain starts at the Jev-failure model."""
        out = [self.jev_failure_model] if cls is None else [self.classes[cls]["primary"], self.classes[cls]["long_context"]]
        while out[-1] in self.long_context_next and self.long_context_next[out[-1]] not in out:
            out.append(self.long_context_next[out[-1]])
        return list(dict.fromkeys(out))


def load_questions(path: str | Path, sha256: str, dimension: str) -> dict:
    """The `questions` field of a Jev request, built from the prompt file (which must match its pinned sha256)."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{path} does not match the sha256 pinned in the policy")
    doc = json.loads(raw)
    q = doc["questions"][dimension]
    return {dimension: {"type": "choice", "instructions": doc["preamble"] + "\n\n" + q["instructions"],
                        "criteria": dict(q["criteria"])}}


# ---- what Jev is shown ---------------------------------------------------------------------------------------

TOOL_LIST_BUDGET = 3_000        # estimated tokens; a longer compact tool list is shown as counts per category
_CJK = re.compile(r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff00-\uffef]")
_CATEGORY_RULES = (
    (("bash", "shell", "exec", "terminal", "run_command", "command"), "command_execution"),
    (("git", "github", "gitlab", "repo", "pull_request"), "repository_operations"),
    (("file", "read", "write", "edit", "str_replace", "ls", "glob", "grep", "directory"), "file_operations"),
    (("browser", "web", "fetch_url", "search_web", "navigate", "http"), "web_browsing"),
    (("sheet", "excel", "airtable", "table"), "spreadsheet_operations"),
    (("calendar", "event", "meeting"), "calendar_operations"),
    (("slack", "teams", "discord", "gmail", "email", "mail", "sms", "message"), "messaging"),
    (("salesforce", "hubspot", "crm", "pipedrive", "zoho", "lead", "deal", "opportunit"), "crm_operations"),
    (("jira", "asana", "trello", "linear", "ticket", "zendesk", "freshdesk", "issue"), "ticketing_project"),
    (("stripe", "quickbooks", "xero", "invoice", "payment"), "finance_operations"),
    (("sql", "database", "db_", "query"), "database"),
    (("api_search", "api_fetch", "search_tools", "api"), "api_access"),
)


def estimate_tokens(obj) -> int:
    """Approximate size in tokens of a string or JSON value: CJK characters 1.1 each, digits 1 each, other
    characters 1 per 2.5. Used only to decide whether the compact tool list is short enough to show in full."""
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    cjk = len(_CJK.findall(s))
    digits = sum(ch.isdigit() for ch in s)
    return round(cjk * 1.1 + digits + (len(s) - cjk - digits) / 2.5)


def tool_category(name: str) -> str:
    n = name.lower()
    for keys, category in _CATEGORY_RULES:
        if any(k in n for k in keys):
            return category
    return (n.split("_")[0] or "other") + "_operations"


def _first_sentence(text: str, limit: int = 160) -> str:
    s = text.strip().split("\n")[0]
    return re.split(r"(?<=[.!?\u3002])\s", s, maxsplit=1)[0][:limit]


def compact_tools(tools) -> list[dict]:
    """Each named tool as {name, category, first sentence of its description}; parameter schemas are dropped."""
    out = []
    for t in tools or ():
        if isinstance(t, dict):
            f = t["function"] if isinstance(t.get("function"), dict) else t
            name, desc = str(f.get("name") or ""), str(f.get("description") or "")
        else:
            name, desc = (str(t), "") if t else ("", "")
        if name:
            out.append({"name": name, "category": tool_category(name), "description": _first_sentence(desc)})
    return out


def tool_environment(tools) -> tuple[object, str]:
    """(representation, "none" | "compact" | "aggregated"): the compact tool list, or {category: tool count}
    when that list is longer than TOOL_LIST_BUDGET."""
    compact = compact_tools(tools)
    if not compact:
        return None, "none"
    if estimate_tokens(compact) <= TOOL_LIST_BUDGET:
        return compact, "compact"
    return dict(sorted(Counter(t["category"] for t in compact).items())), "aggregated"


def _view_message(m: dict) -> dict:
    content, texts, other = m.get("content"), [], 0
    if isinstance(content, str):
        texts = [content]
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text") or "")
            elif part is not None:
                other += 1
    elif content is not None:
        texts = [str(content)]
    out = {"role": m.get("role"), "content": "\n".join(texts)}
    if other:
        out["non_text_parts"] = other
    return out


def build_view(messages: list, tools=None) -> dict:
    """The state sent to Jev: every message's role and full text (text parts joined by newlines, plus the number of
    non-text parts such as images) and the tools in compact form. Neither argument is modified.

    Jev's input size is limited. For requests too long for Jev's input, supply your own condensed view through
    Router(build_view=...). The selected model always receives the original request."""
    view = {"messages": [_view_message(m) for m in messages or () if isinstance(m, dict)]}
    env, kind = tool_environment(tools)
    if env is not None:
        view["tools" if kind == "compact" else "tool_categories"] = env
    return view


# ---- Jev -----------------------------------------------------------------------------------------------------

class JevError(Exception):
    pass


def select_class(answer: dict) -> str:
    """The class with the highest probability (ties: the one listed first); Jev's `choice` without probabilities."""
    probs = {str(k): float(v) for k, v in (answer.get("probabilities") or {}).items() if isinstance(v, (int, float))}
    return max(probs, key=probs.get) if probs else str(answer["choice"])


class JevClient:
    """Asks Jev for the class of a conversation. The API key is read from JEV_API_KEY unless given."""

    def __init__(self, questions: dict, *, api_key: str | None = None, url: str = JEV_URL, attempts: int = 4,
                 backoff_s: float = 1.0, timeout_s: float = 30.0, client: httpx.Client | None = None):
        self.questions, self.dimension = questions, next(iter(questions))
        self.api_key = api_key if api_key is not None else os.environ.get("JEV_API_KEY")
        self.url, self.attempts, self.backoff_s = url, max(1, attempts), backoff_s
        self.client = client or httpx.Client(timeout=timeout_s)

    @classmethod
    def from_policy(cls, policy: Policy, **kwargs) -> "JevClient":
        return cls(load_questions(policy.prompt_path, policy.prompt_sha256, policy.prompt_dimension), **kwargs)

    def request_body(self, view: dict) -> dict:
        return {"model": JEV_MODEL, "state": view, "questions": self.questions}

    def classify(self, view: dict, class_names) -> tuple[str, dict]:
        """-> (class, Jev's answer). A failed call or a class outside `class_names` is tried again; JevError when
        every attempt fails."""
        if not self.api_key:
            raise JevError("no Jev API key (set JEV_API_KEY)")
        last = None
        for i in range(self.attempts):
            try:
                r = self.client.post(self.url, json=self.request_body(view),
                                     headers={"authorization": f"Bearer {self.api_key}"})
                if r.status_code != 200:
                    raise JevError(f"HTTP {r.status_code}")
                answer = r.json()["answers"][self.dimension]
                cls = select_class(answer)
                if cls not in class_names:
                    raise JevError(f"unknown class {cls!r}")
                return cls, answer
            except (httpx.HTTPError, JevError, KeyError, ValueError, TypeError, AttributeError) as e:
                last = e
                if i < self.attempts - 1:
                    time.sleep(self.backoff_s * 2 ** i)      # 1, 2, 4 s with the defaults
        raise JevError(f"Jev failed: {type(last).__name__}: {last}")


# ---- sessions ------------------------------------------------------------------------------------------------

def _text(content) -> str:
    """Text parts of a message's content, concatenated."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join((p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return "" if content is None else str(content)


def session_key(messages: list, session_id: str | None = None) -> str | None:
    """A client-supplied session id when given; otherwise sha256(first system/developer text + NUL + first user
    text)[:32], which stays the same across the turns of a conversation. None when there is no user message."""
    if session_id:
        return str(session_id)
    system, first_user = "", None
    for m in messages or ():
        if not isinstance(m, dict):
            continue
        if m.get("role") in ("system", "developer") and not system:
            system = _text(m.get("content"))
        elif m.get("role") == "user" and first_user is None:
            first_user = _text(m.get("content"))
    if first_user is None:
        return None
    h = hashlib.sha256(system.encode("utf-8", "replace") + b"\x00" + first_user.encode("utf-8", "replace"))
    return h.hexdigest()[:32]


@dataclass
class Session:
    cls: str | None                 # None when Jev failed
    kind: str | None
    chain: list[str]
    model: str
    jev_failure: bool = False
    answer: dict = field(default_factory=dict)
    moves: list = field(default_factory=list)      # (from, to, reason)
    last_seen: float = 0.0

    def move_to(self, model: str, reason: str) -> bool:
        """Move forward along the chain to `model`; a session never moves back."""
        if self.model in self.chain and model in self.chain and self.chain.index(model) > self.chain.index(self.model):
            self.moves.append((self.model, model, reason))
            self.model = model
            return True
        return False


class SessionStore:
    """In memory, with sliding expiry: a session unused for `idle_expiry_s` seconds is dropped and the next request
    of that conversation is classified again. Beyond `max_sessions`, the least recently used session is dropped."""

    def __init__(self, idle_expiry_s: float = 6 * 3600, max_sessions: int = 200_000,
                 clock: Callable[[], float] = time.time):
        self.idle_expiry_s, self.max_sessions, self.clock = idle_expiry_s, max_sessions, clock
        self._d: OrderedDict[str, Session] = OrderedDict()

    def get(self, key: str) -> Session | None:
        s = self._d.get(key)
        if s is None:
            return None
        now = self.clock()
        if now - s.last_seen > self.idle_expiry_s:
            del self._d[key]
            return None
        s.last_seen = now
        self._d.move_to_end(key)
        return s

    def put(self, key: str, s: Session) -> None:
        s.last_seen = self.clock()
        self._d[key] = s
        self._d.move_to_end(key)
        while len(self._d) > self.max_sessions:
            self._d.popitem(last=False)


# ---- handoff -------------------------------------------------------------------------------------------------

def _first_choice(reply) -> dict:
    try:
        choice = reply["choices"][0]
    except (KeyError, IndexError, TypeError):
        return {}
    return choice if isinstance(choice, dict) else {}


def answers(message) -> bool:
    """A message answers when it has non-blank content or tool calls; reasoning text alone does not."""
    return isinstance(message, dict) and bool(_text(message.get("content")).strip() or message.get("tool_calls")
                                              or message.get("function_call"))


def is_failed_reply(reply) -> bool:
    """A completion without choices or with finish_reason "error" is a failed request, retried like any other."""
    choice = _first_choice(reply)
    return not choice or choice.get("finish_reason") == "error"


def needs_handoff(reply=None, error: bool = False) -> str | None:
    """Trigger for a finished non-streamed request, or None: "error", "length" or "empty".
    reply: the parsed body of the final 2xx completion. error: the request still failed after retrying on the same
    model (network errors, timeouts, rate limiting, server errors). Client errors such as an invalid request do not
    lead to a handoff, and an input too long for the model moves along the chain instead (Router.on_input_too_long)."""
    if error or is_failed_reply(reply):
        return "error"
    choice = _first_choice(reply)
    if choice.get("finish_reason") == "length":
        return "length"
    if not answers(choice.get("message")):
        return "empty"
    return None


def handoff_target(policy: Policy, session: Session | None, served_model: str, reply=None, *,
                   error: bool = False, stream: bool = False) -> str | None:
    """The model that answers the same request again, or None. Only for the kinds the handoff covers (direct
    classes), only for non-streamed requests, and only when the handoff model is ahead of `served_model` on the
    session's chain (so never when the handoff model itself answered)."""
    h = policy.handoff
    if not h or stream or session is None or session.kind not in h["kinds"] or served_model == h["model"]:
        return None
    chain, to = session.chain, h["model"]
    if served_model not in chain or to not in chain or chain.index(to) <= chain.index(served_model):
        return None
    return to if needs_handoff(reply, error) in h["triggers"] else None


# ---- router --------------------------------------------------------------------------------------------------

def always_fits(model: str, request: dict) -> bool:
    return True


@dataclass
class Decision:
    model: str | None               # None: no model on the session's chain holds the request
    cls: str | None
    kind: str | None
    session_key: str | None
    session: Session
    new_session: bool               # Jev was asked: first request of the session
    jev_failure: bool
    moved: bool = False             # the session moved along its chain on this request
    request: dict = field(default_factory=dict, repr=False)     # {"messages", "tools"}, as passed to fits


class Router:
    """fits(model, request) -> bool says whether `model` can take `request` ({"messages", "tools"}). Whether a
    request fits depends on the provider; set it according to your provider. By default every request fits."""

    def __init__(self, policy: Policy, jev: JevClient | None = None, *, fits: Callable[[str, dict], bool] = always_fits,
                 build_view: Callable[[list, object], dict] = build_view, store: SessionStore | None = None):
        self.policy, self.fits, self.build_view = policy, fits, build_view
        self.jev = jev or JevClient.from_policy(policy)
        self.store = store or SessionStore()
        self._locks: dict[str, threading.Lock] = {}
        self._mutex = threading.Lock()

    def _lock(self, key: str | None):
        if key is None:
            return threading.Lock()
        with self._mutex:
            return self._locks.setdefault(key, threading.Lock())

    def _forward_fit(self, s: Session, request: dict, reason: str) -> str | None:
        """Move s to the first later model on its chain that holds the request; None if there is none."""
        start = s.chain.index(s.model) + 1 if s.model in s.chain else 1
        for model in s.chain[start:]:
            if self.fits(model, request):
                s.moves.append((s.model, model, reason))
                s.model = model
                return model
        return None

    def route(self, messages: list, tools=None, session_id: str | None = None) -> Decision:
        """The model for this request. Jev is asked only when the request's session does not exist yet."""
        key = session_key(messages, session_id)
        request = {"messages": messages, "tools": tools}
        with self._lock(key):
            s = self.store.get(key) if key else None
            if s is not None:
                if self.fits(s.model, request):
                    return Decision(s.model, s.cls, s.kind, key, s, False, s.jev_failure, request=request)
                model = self._forward_fit(s, request, "input_size")
                return Decision(model, s.cls, s.kind, key, s, False, s.jev_failure, model is not None, request)
            try:
                cls, answer = self.jev.classify(self.build_view(messages, tools), self.policy.class_names)
            except JevError:
                cls, answer = None, {}
            chain = self.policy.chain(cls)
            model = next((m for m in chain if self.fits(m, request)), None)
            s = Session(cls, self.policy.kind(cls), chain, model or chain[-1], jev_failure=cls is None, answer=answer)
            if model is not None and model != chain[0]:
                s.moves.append((chain[0], model, "input_size"))
            if key:
                self.store.put(key, s)
            return Decision(model, cls, s.kind, key, s, True, cls is None, model not in (None, chain[0]), request)

    def on_input_too_long(self, decision: Decision) -> str | None:
        """The provider rejected the request as too long for `decision.model`: the next model on the chain that
        holds it (the session moves there), or None."""
        with self._lock(decision.session_key):
            s = decision.session
            if s.model != decision.model:          # another request of the session already moved it forward
                return s.model
            return self._forward_fit(s, decision.request, "input_size")

    def handoff(self, decision: Decision, reply=None, *, error: bool = False, stream: bool = False,
                served_model: str | None = None) -> str | None:
        """After a request answered by `served_model` (default decision.model): the model to send the same request
        to once, or None (see handoff_target). When a model is returned, the session moves to it, so later requests
        of the session go there too. No handoff when that model cannot hold the request."""
        to = handoff_target(self.policy, decision.session, served_model or decision.model, reply,
                            error=error, stream=stream)
        if to is None or not self.fits(to, decision.request):
            return None
        with self._lock(decision.session_key):
            decision.session.move_to(to, "handoff_" + str(needs_handoff(reply, error)))
        return to
