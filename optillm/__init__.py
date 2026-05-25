# Version information
__version__ = "0.3.15"

_LAZY_EXPORTS = {
    "main": (".server", "main"),
    "server_config": (".server", "server_config"),
    "app": (".server", "app"),
    "known_approaches": (".server", "known_approaches"),
    "plugin_approaches": (".server", "plugin_approaches"),
    "parse_combined_approach": (".server", "parse_combined_approach"),
    "parse_conversation": (".server", "parse_conversation"),
    "extract_optillm_approach": (".server", "extract_optillm_approach"),
    "get_config": (".server", "get_config"),
    "load_plugins": (".server", "load_plugins"),
    "count_reasoning_tokens": (".server", "count_reasoning_tokens"),
    "parse_args": (".server", "parse_args"),
    "execute_single_approach": (".server", "execute_single_approach"),
    "execute_combined_approaches": (".server", "execute_combined_approaches"),
    "execute_parallel_approaches": (".server", "execute_parallel_approaches"),
    "generate_streaming_response": (".server", "generate_streaming_response"),
    "attach_optillm_routing": (".approach_router.resolver", "attach_optillm_routing"),
    "ApproachPredictor": (".approach_router", "ApproachPredictor"),
    "RoutingDecision": (".approach_router", "RoutingDecision"),
    "ROUTING_MODES": (".approach_router", "ROUTING_MODES"),
    "resolve_optillm_approach": (".approach_router.resolver", "resolve_optillm_approach"),
}

__all__ = ["__version__", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    import importlib

    module = importlib.import_module(module_name, __package__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
