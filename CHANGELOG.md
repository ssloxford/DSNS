# Changelog

## Version 1.1.0

This release comprises a series of performance improvements and bug fixes. Behaviour is largely unchanged except in a few edge cases, listed below.

### Changed behaviour

These changes can alter results or output. Check whether they affect you.

- **State is no longer shared between simulations run in one process.** Several classes created their lists, dicts and sets at class level, so every instance shared one container. Multiple simulations running in the same process would see each other's events and messages. For example, a second `LoggingActor` started with the first one's `events`, and a second `EarthMoonMarsMultiConstellation` had 650 Earth nodes instead of 325. Affected classes: `LoggingActor`, `PreprocessedLoggingActor`, `MessageRoutingActor`, `MessageBroadcastActor`, `LookaheadRoutingDataProvider`, `MessageLocationTracker`, `LTPConfig`, `LinkLossProbabilityUpdateEvent`, `MultiConstellationVisualizer`, and the `EarthMoon`, `EarthMars` and `EarthMoonMars` multi-constellation presets.
  - **Who is affected:** anyone running more than one simulation per process (notebooks, parameter sweeps, test suites). Results from the second and later runs may differ from 1.0.x, and the new results are the correct ones. A single run in a fresh process gives identical logs.
  - **What to do:** nothing, but re-run any multi-run experiments that used 1.0.x. Attributes such as `LoggingActor.events` now exist on instances only, not on the class.
  - Message `uid` values come from one process-wide generator and are deliberately still shared, so back-to-back runs get different uids. Compare logs without uids.
- **A message's creation time is now set by `MessageCreatedEvent`.** `MessageCreatedEvent.__init__` sets `message.time` to the event time. Previously the routing actor set it after logging actors had already seen the message. Because the time is part of the message hash, `PreprocessedLoggingActor` stored the message under its old key. Messages created with the default `time=0.0` (for example by `MultiPointToPointTrafficActor`) therefore had wrong start times and a stale, undelivered duplicate entry.
  - **Who is affected:** anyone reading `PreprocessedLoggingActor` output. Start times (and so latencies) are now correct, and the stale duplicates no longer appear, so entry counts can drop.
  - **What to do:** Nothing. Delivery and drop events (times and hop counts) are unchanged. If you wrote your own creation event, set the message time in the same way.

### Added

- `MultiConstellation.get_delays_batch(pairs)`: vectorised equivalent of calling `get_delay` for each pair. It returns an array in the order of `pairs` and raises `ValueError` for unknown satellite IDs and `RuntimeError` if called before `update_positions()`.
- `PreprocessedLoggingActor(created_event_types=...)`: the event types that signal message creation (default `(MessageCreatedEvent,)`). Extensions with their own creation events pass them here. Each type must have a `message` attribute.
- `WalkerISLHelper(phase_offset=0.0)`: opt-in shift of cross-seam links for Walker constellations with a phase offset between planes. The shift is `round(num_planes * phase_offset * sats_per_plane / 360) mod sats_per_plane` slots. The default of `0.0` gives the same links as before and no preset passes it.
- A small test suite under `tests/` (see Internal).

### Changed

- `BestEffortRoutingDataProvider.update` sets edge weights in place when the link list is identical to the previous update (same pairs, same order, no duplicates). Otherwise it rebuilds the graph as before. Next hops, distances and path costs are identical to a rebuild. The saving is small (about 0.4 ms per update).
- `WalkerConstellation.update_positions` uses a closed-form calculation for exactly circular orbits (`eccentricity == 0`) and the previous Kepler calculation otherwise. Positions agree to within 7.7e-8 m. The positions array is updated in place and each satellite keeps its own copy of its position.
- `LinkTransmissionActor` and `LTPMessageRetransmissionActor`: `message_location_tracker` now defaults to `None` (see Deprecated).
- `BestEffortRoutingDataProvider` and `LookaheadRoutingDataProvider` compute link weights with `MultiConstellation.get_delays_batch` instead of calling `get_delay` per link. If the mobility model overrides `get_delay` or `get_distance`, they fall back to per-link `get_delay`, so custom delay models are still used. See Performance.

### Deprecated

- Creating `LinkTransmissionActor` or `LTPMessageRetransmissionActor` without a `message_location_tracker` emits a `DeprecationWarning`. It still works, using a shared default tracker, which carries state between simulations in the same process. Create one `MessageLocationTracker` per simulation and pass it to both actors. The argument will be required in the next major version.
- `PreprocessedLoggingActor` still detects `dsns.key_management.UnsignedMessageCreatedEvent` automatically if that module is importable, but now emits a `DeprecationWarning`. Pass `created_event_types=(MessageCreatedEvent, UnsignedMessageCreatedEvent)` instead.

### Fixed

- Shared mutable state between instances and simulations (see Changed behaviour).
- `PreprocessedLoggingActor` lost or duplicated messages created without a time (see Changed behaviour).
- `LookaheadRoutingDataProvider.initialize()` started from the previous provider's graphs, so a second provider held twice as many and rotated them incorrectly. It now starts from an empty set.
- `dsns.message_actors` no longer tries to import `AttackStrategy` from `dsns.key_management`; it uses the `AttackStrategy` type from `dsns.message`, which the external KeySpace enum of the same name used to shadow.

### Performance

- `WalkerConstellation.update_positions` is about 40 to 45 times faster for circular orbits. For 200 updates: Iridium (66 satellites) 0.55 s to 0.012 s; a Starlink-like 72x22 constellation (1584 satellites) 14.1 s to 0.35 s.
- `MultiConstellation.get_delays_batch` is about 10x (cold cache) to 23x (warm cache) faster than looping over `get_delay`, for 2000 links in a 1000-satellite Walker constellation (11.7 ms to 1.2 ms and 0.5 ms).
- Routing data providers compute link weights with `get_delays_batch`: `BestEffortRoutingDataProvider.update` is about 7 times faster (Iridium, 609 links: 7.8 ms to 1.2 ms per update; Starlink, 5374 links: 64 ms to 8.4 ms). The batched delays were bit-identical to `get_delay` across about 38,000 links on four presets, so routes are unchanged.
- Best-effort in-place weight updates: see Changed. Measured gain is negligible.

### Packaging

- Documentation tools (`sphinx`, `sphinx-argparse`, `sphinx-rtd-theme`) moved from the required dependencies to a `docs` extra: `pip install "dsns[docs]"`. A plain `pip install dsns` no longer installs Sphinx.
- `trimesh` and `pillow`, which the visualiser imports, are now declared as dependencies.
- `pandas` removed from the dependencies. It was declared but never imported by the package. The plotting notebooks in `examples/paper` need `pandas` and `matplotlib` installed separately, as noted in the README.
- Removed an invalid `pyrender; python_version<="0.1.45"` requirement (a plain `pyrender` requirement remains).
- Package data (`data/*.txt` and the web visualiser HTML, CSS and JavaScript files) is declared explicitly and verified to be in the built wheel.

### Internal

- Added a `pytest` suite (129 tests on the release candidate): geometry, multi-constellation, Walker positions and seams, batched delays, routing weights, shared state, extension hooks, packaging, message creation time, tracker deprecation, and a 1 s smoke simulation. Run it with `pytest tests`.
- New GitHub Actions workflow `tests.yml` runs the suite on Python 3.11 and 3.12. It has not yet run on GitHub.
- The documentation workflow installs `".[docs]"`. The `docs` extra requires `sphinx-argparse>=0.6`, which resolves `:filename:` paths relative to `docs/`; the example pages now use `../examples/...`.
