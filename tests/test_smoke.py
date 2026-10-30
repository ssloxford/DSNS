"""End-to-end smoke test: tiny Walker constellation, routing, logging, short run, messages arrive."""
import numpy as np

from dsns.constellation import WalkerConstellation, WalkerISLHelper
from dsns.helpers import EARTH_RADIUS, IDHelper
from dsns.logging import BandwidthLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.multiconstellation import MultiConstellation
from dsns.simulation import Simulation
from dsns.traffic_sim import MultiPointToPointTrafficActor
from dsns.transmission import LinkTransmissionActor, MessageLocationTracker

MESSAGE_INTERVAL = 20.0
SIM_TIME = 300.0


def _build(store_and_forward: bool):
    ids = IDHelper()
    walker = WalkerConstellation(
        name="smoke", num_planes=3, sats_per_plane=6, inclination=53.0,
        semi_major_axis=EARTH_RADIUS + 550e3, eccentricity=1e-32, ascending_node_arc=360.0, phase_offset=0.0,
        isl_helper=WalkerISLHelper(num_planes=3, sats_per_plane=6, intra_layer_links=True, inter_layer_links=0),
        id_helper=ids,
    )
    constellation = MultiConstellation()
    constellation.duration = SIM_TIME
    constellation.add_constellation(walker)
    sat_ids = list(walker.satellites.ids)

    tracker = MessageLocationTracker()
    transmission = LinkTransmissionActor(default_bandwidth=25e6 // 8, buffer_if_link_busy=True,
                                         reroute_on_link_down=True, message_location_tracker=tracker)
    # sat 0 -> sat 9 and sat 4 -> sat 15: a few hops apart, 1 MB every 20 s
    config = [("T0", sat_ids[0], sat_ids[9], 8e6, MESSAGE_INTERVAL), ("T1", sat_ids[4], sat_ids[15], 8e6, MESSAGE_INTERVAL)]
    traffic = MultiPointToPointTrafficActor(config, update_interval=60, cutoff=200)
    if store_and_forward:
        provider = LookaheadRoutingDataProvider(resolution=15.0, num_steps=20)
    else:
        provider = BestEffortRoutingDataProvider()
    routing = MessageRoutingActor(provider, store_and_forward=store_and_forward, model_bandwidth=True)
    logger = PreprocessedLoggingActor(log_other=False)
    bandwidth = BandwidthLoggingActor()

    sim = Simulation(constellation, actors=[transmission, traffic, routing], logging_actors=[logger, bandwidth],
                     data_providers=[provider], timestep=15.0)
    return sim, logger, bandwidth, constellation


def _run(store_and_forward: bool):
    sim, logger, bandwidth, constellation = _build(store_and_forward)
    sim.initialize(time=0)
    sim.run(SIM_TIME, progress=False)
    return logger, bandwidth, constellation


def _assert_delivered(logger):
    messages = list(logger.direct_messages.values())
    assert len(messages) > 0, "traffic actor generated no messages"
    delivered = [m for m in messages if m.delivered]
    assert len(delivered) > 0, "no message was delivered"
    for m in delivered:
        assert m.end_time >= m.start_time
        # a few hops of ~1000-5000 km plus 1 MB at 25 Mbit/s: seconds, never minutes
        assert m.end_time - m.start_time < 120.0
        assert m.hops is not None and m.hops >= 1
        assert not m.dropped
    return delivered


def test_best_effort_delivery():
    logger, bandwidth, constellation = _run(store_and_forward=False)
    _assert_delivered(logger)


def test_store_and_forward_delivery_and_bandwidth_log():
    logger, bandwidth, constellation = _run(store_and_forward=True)
    _assert_delivered(logger)
    assert len(bandwidth.transmissions) > 0
    assert all(np.isfinite(v) for v in constellation.satellite_positions.ravel())
