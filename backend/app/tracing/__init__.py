from __future__ import annotations

import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

_TRACES: OrderedDict[str, dict] = OrderedDict()
_MAX_TRACES = 1000


@dataclass
class Trace:
    id: str
    name: str
    inputs: dict
    events: list[dict] = field(default_factory=list)

    def log(self, step: str, data: Any) -> None:
        self.events.append({"step": step, "data": data})


@contextmanager
def start_trace(name: str, inputs: dict):
    t = Trace(id=str(uuid.uuid4()), name=name, inputs=inputs)
    try:
        yield t
    finally:
        _TRACES[t.id] = {"id": t.id, "name": t.name, "inputs": t.inputs, "events": t.events}
        if len(_TRACES) > _MAX_TRACES:
            _TRACES.popitem(last=False)


def get_trace(trace_id: str) -> dict | None:
    return _TRACES.get(trace_id)
