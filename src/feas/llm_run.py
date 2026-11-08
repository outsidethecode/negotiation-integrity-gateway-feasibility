"""Run one LLM configuration over a scenario set (shared by scripts/run_llm.py
and the harness tests). The run script owns the TrustedSender; the seller gets
only the public session view (and, in D-preview, the version-bound preview)."""
from .llm_agent import LLMSeller
from .preview import Preview
from .session import run_config

INTERFACES = {"C": False, "D": True, "Dp": True}  # interface -> behind the gate?


def run_llm_config(cfg, model, scenarios, interface, sender):
    uses_gate = INTERFACES[interface]
    sellers = []

    def factory(ep, sid, ledger):
        sender.set_secrets(ep["canary"], repr(tuple(ep["mandate"])))
        seller = LLMSeller(interface, sender, cfg["round_limit"], cfg["seller_public"],
                           cap_units=cfg["seller_public"]["quantity_cap"])
        if interface == "Dp":
            seller.set_preview(Preview(model, ledger).snapshot)
        sellers.append((sid, seller))
        return seller

    run = run_config(cfg, model, scenarios, f"{interface}_llm", uses_gate=uses_gate, seller_factory=factory)
    calls = [dict(c, session=sid) for sid, s in sellers for c in s.calls]
    return run, calls
