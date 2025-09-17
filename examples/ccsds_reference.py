import os
import pickle
import random
import datetime
from typing import Optional
import argparse

from dsns.logging import BandwidthLoggingActor, PreprocessedLoggingActor, LTPTransmissionLoggingActor
from dsns.message import Link, LossConfig, LTPConfig
from dsns.simulation import Simulation, LoggingActor
from dsns.transmission import (
    MessageLocationTracker,
    LinkTransmissionActor,
    LTPMessageRetransmissionActor,
    RetransmissionConfig,
)
from dsns.message_actors import (
    MessageRoutingActor,
    BestEffortRoutingDataProvider,
    LookaheadRoutingDataProvider,
)
from dsns.traffic_sim import MultiPointToPointTrafficActor
from dsns.presets import EarthMoonMarsMultiConstellation

class EarthMoonLossConfig(LossConfig):
    def __init__(self, seed: float = 0, default_loss_probability: float = 0.005, max_frame_size: int = 64 * 1024):
        super().__init__(seed, default_loss_probability)
        self.max_frame_size = max_frame_size

    def is_message_lost(self, source: int, destination: int, size: int) -> bool:
        link = Link(source=source, destination=destination)
        rng = self._get_rng_for_link(link)
        loss_probability = self._get_loss_probability_for_link(link)
        num_frames = (size + self.max_frame_size - 1) // self.max_frame_size

        for _ in range(num_frames):
            if rng.random() < loss_probability:
                return True
        return False

def main(scenario: str = "earth-lunar", delivery: str = "store_and_forward", logging: bool = True, results_dir: str = "earth-moon-results", verbose:bool = False, frame_size: int = 64 * 1024, loss: float = 0.005):
    constellation = EarthMoonMarsMultiConstellation(
        moon=True,
        mars=True,
        moon_mars_link=False
    )

    earth_nodes = constellation.earth_nodes
    moon_nodes = constellation.moon_nodes

    source = 0
    destination = 345 if scenario == "earth-lunar" else 424

    message_config = [("EarthToMoon", source, destination, 1e6, 15.0)]

    actors = []
    data_providers = []
    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=100e6 / 8,  # 100 Mbps
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker
    )
    actors.append(transmission_actor)

    traffic_actor = MultiPointToPointTrafficActor(
        message_config,
        update_interval=300,
        reliable_messages=(delivery == "ltp"),
        cutoff=60 * 60 * 8,
    )
    actors.append(traffic_actor)

    loss_config = EarthMoonLossConfig(default_loss_probability=loss, max_frame_size=frame_size) if delivery != "ltp" else LossConfig(seed=0, default_loss_probability=loss)

    if delivery == "best_effort":
        routing_data_provider = BestEffortRoutingDataProvider()
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=False,
            model_bandwidth=True,
            loss_config=loss_config,
        )
    elif delivery == "store_and_forward":
        routing_data_provider = LookaheadRoutingDataProvider(resolution=15.0, num_steps=600)
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=True,
            model_bandwidth=True,
            loss_config=loss_config,
        )
    elif delivery == "ltp":
        routing_data_provider = LookaheadRoutingDataProvider(resolution=15.0, num_steps=600)
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=True,
            model_bandwidth=True,
            loss_config=loss_config,
            reliable_transfer_config=LTPConfig(max_segment_size=frame_size),
        )
        ltp_actor = LTPMessageRetransmissionActor(
            config=RetransmissionConfig(max_retries=100),
            model_bandwidth=True,
            message_location_tracker=message_location_tracker
        )
        actors.append(ltp_actor)
    else:
        raise ValueError(f"Unsupported delivery type: {delivery}")

    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    if logging:
        pre = PreprocessedLoggingActor(log_other=False)
        bw = BandwidthLoggingActor()
        ltp = LTPTransmissionLoggingActor()
        logging_actors = [pre, bw, ltp]
    else:
        logging_actors = []

    sim = Simulation(
        constellation,
        actors=actors,
        logging_actors=logging_actors + ([LoggingActor(verbose=True)] if verbose else []),
        data_providers=data_providers,
        timestep=15.0
    )

    sim.initialize(0)
    sim.run(60 * 60 * 12, progress=True)

    if logging:
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f"{scenario}-{delivery}.pickle"), "wb") as f:
            pickle.dump((pre.direct_messages, pre.broadcast_messages, pre.other_events), f)

        with open(os.path.join(results_dir, f"{scenario}-{delivery}-bw.pickle"), "wb") as f:
            pickle.dump(bw.aggregate(1.0, default_bandwidth=100e6 / 8), f)

        with open(os.path.join(results_dir, f"{scenario}-{delivery}-ltp.pickle"), "wb") as f:
            pickle.dump(ltp.aggregate(1.0), f)

def get_parser():
    parser = argparse.ArgumentParser(description="Run a CCSDS reference scenario in DSNS.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["earth-lunar", "earth-mars"],
        default="earth-lunar",
        help="The scenario to run"
    )
    parser.add_argument(
        "--delivery",
        type=str,
        choices=["best_effort", "store_and_forward", "ltp"],
        default="store_and_forward",
        help="The delivery method to use"
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    print(f"doing: {args.delivery}")
    main(scenario=args.scenario, delivery=args.delivery, logging=True, results_dir="paper-preparation-results-15s")

"""
parallel -j 6 ::: \
    "python paper_reference.py --delivery best_effort --scenario earth-lunar" \
    "python paper_reference.py --delivery store_and_forward --scenario earth-lunar" \
    "python paper_reference.py --delivery ltp --scenario earth-lunar" \
    "python paper_reference.py --delivery best_effort --scenario earth-mars" \
    "python paper_reference.py --delivery store_and_forward --scenario earth-mars" \
    "python paper_reference.py --delivery ltp --scenario earth-mars"
"""