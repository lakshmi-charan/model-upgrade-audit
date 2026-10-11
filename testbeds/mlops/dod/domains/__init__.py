"""Domain registry."""
from importlib import import_module

DOMAINS = ("acs", "tlc", "news", "divvy", "hmda", "bts")


def get_pipeline(domain: str):
    return import_module(f"dod.domains.{domain}").PIPELINE


def get_domain(domain: str):
    return import_module(f"dod.domains.{domain}")
