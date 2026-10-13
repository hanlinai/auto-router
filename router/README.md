# Auto routing decision

This directory holds the routing decision used for every Auto run in the report, as a small reference
implementation in one file, `jev_router.py`. For each new conversation, Jev picks one of 16 task classes, the policy
in `policy.json` maps the class to a model, and every later request of the session stays on that model. When the
model of a direct-answer class (in the report's policy, DS 4.1, or DS 0731 for long table requests) gives no answer (blank content, reasoning text only, output cut off at the maximum
length, or a request that still fails after retries on the same model), the same request is sent once to Kimi K3 and
the session moves to Kimi K3; agent classes are only retried on the same model. A request that outgrows its model
moves forward along the class's long-context chain and never moves back. Sections 2.1 to 2.5 of the report describe
these rules.

The module decides which model serves a request. Forwarding the request, retrying failures on the same model,
streaming and logging belong to the gateway that uses it.

| File | Contents |
|---|---|
| `jev_router.py` | Policy loading, the view Jev is shown, the Jev call and class selection, session keys and the session store, the long-context chain walk, the handoff rule |
| `policy.json` | The policy used in the report: 16 classes with their kind, primary model and long-context model, the long-context chain step, the handoff rule and the model used when Jev fails |
| `jev_prompt.json` | The Jev prompt used in the report (reproduced in the report's appendix); its sha256 is pinned in `policy.json` |
| `tests/` | Unit tests with a fake Jev; no network access |

## Usage

```python
import jev_router as jr

router = jr.Router(jr.Policy.load("policy.json"))        # Jev API key from JEV_API_KEY

decision = router.route(body["messages"], body.get("tools"), session_id=client_session_id)
body["model"] = decision.model           # the selected model receives the original request
reply = send(body)                       # your client: retry failures on the same model

to = router.handoff(decision, reply)     # error=True if the request still failed after those retries
if to:                                   # pass stream=True for streamed requests: they are never handed off
    body["model"] = to
    reply = send(body)                   # the client gets this reply; later requests of the session go to `to`
```

`decision.model` is `None` when no model on the chain can take the request. `decision.cls`, `decision.new_session`
(Jev was asked) and `decision.session.answer` (Jev's answer with the probabilities of all classes) are there for
logging.

## What is configurable

- **Policy** (`policy.json`): the classes and their models, the long-context chain, the handoff model, kinds and
  triggers, the model used when Jev fails, and the prompt file with its sha256. In the report's runs, routed
  requests also dropped the client's temperature and top_p, so each model ran at its provider's default sampling.
- **Capacity** (`Router(fits=...)`): `fits(model, request)` says whether a model can take a request
  (`{"messages", "tools"}`). This depends on the provider; set it according to your provider. By default every
  request fits, so a session moves along its chain only when `fits` says so or when you call
  `router.on_input_too_long(decision)` after the provider rejects a request as too long.
- **What Jev is shown** (`Router(build_view=...)`): by default every message in full plus a compact tool list
  (name, category and first sentence of each description; category counts for a long list). For requests too long
  for Jev's input, supply your own condensed view. The selected model always receives the original request.
- **Jev client** (`JevClient`): API key (`api_key=` or `JEV_API_KEY`), URL (`url=`), attempts with exponential backoff, and timeout. When
  Jev cannot be reached or returns a class that is not in the policy, the session goes to the Jev-failure model.
- **Sessions** (`SessionStore`): a client-supplied session id takes precedence; otherwise the session key is a hash
  of the first system message and the first user message. Sessions expire after `idle_expiry_s` seconds without
  requests, and the next request is classified again.

## Requirements

Python 3.10 or later, `pip install -r requirements.txt`, a Jev API key from TypeSafe AI, and access to the models
named in `policy.json`; in the report, all of them were served by engy.ai. Run the tests with
`python -m pytest tests`.
