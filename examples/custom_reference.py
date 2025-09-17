import os
import pickle
from typing import Optional
import argparse
import random
import datetime

from dsns.logging import BandwidthLoggingActor, LTPTransmissionLoggingActor, PreprocessedLoggingActor
from dsns.message_actors import BestEffortRoutingDataProvider, LookaheadRoutingDataProvider, MessageRoutingActor
from dsns.simulation import Simulation, LoggingActor
from dsns.message import (
    Link,
    LossConfig,
    LTPConfig,
)
from dsns.message_actors import (
    MessageRoutingActor,
    BestEffortRoutingDataProvider,
    LookaheadRoutingDataProvider,
)
from dsns.transmission import (
    LTPMessageRetransmissionActor,
    RetransmissionConfig,
    MessageLocationTracker,
    LinkTransmissionActor,
)
from dsns.presets import (
    IridiumMultiConstellation,
    StarlinkMultiConstellation,
    CubesatMultiConstellation,
    EarthMoonMarsMultiConstellation,
)
from dsns.traffic_sim import (
    MultiPointToPointTrafficActor,
    RandomTrafficActor,
    NormalSampler,
    UniformSampler,
)

class CustomScenarioLossConfig(LossConfig):
    def __init__(self, seed: float = 0, default_loss_probability: float = 0, max_frame_size: int = 64 * 1024):
        super().__init__(seed, default_loss_probability)
        self.max_frame_size = max_frame_size

    def is_message_lost(self, source: int, destination: int, size: int) -> bool:
        link = Link(source=source, destination=destination)
        rng = self._get_rng_for_link(link)
        loss_probability = self._get_loss_probability_for_link(link)
        num_frames = (size + self.max_frame_size - 1) // self.max_frame_size

        for frame in range(num_frames):
            if rng.random() < loss_probability:
                return True
        return False

def main(
        scenario: str = "walker",
        walker_scale: int = 1,
        traffic: str = "point_to_point",
        traffic_scale: int = 10,
        delivery: str = "store_and_forward",
        loss: Optional[float] = None,
        verbose: bool = False,
        results_dir: str = "custom-references",
        logging: bool = True,
    ):
    reliable_messages = delivery == "ltp"
    max_segment_size = 64 * 1024

    if scenario == "walker":
        constellation = IridiumMultiConstellation(iridium_kwargs=dict(
            num_planes=6,
            sats_per_plane=11 * walker_scale,
        ))
        lookahead_resolution = 15.0
    elif scenario == "walker_large":
        constellation = StarlinkMultiConstellation(starlink_kwargs=dict(
            num_planes=72,
            sats_per_plane=22 * walker_scale,
        ))
        lookahead_resolution = 15.0
    elif scenario == "cubesat":
        constellation = CubesatMultiConstellation(
            file_cubesats='assets/cubesats-reference.txt',
            epoch=datetime.datetime(2025, 6, 27, 0, 0, 0)
        )
        lookahead_resolution = 1.0
    elif scenario == "lunar_mars":
        constellation = EarthMoonMarsMultiConstellation(
            moon=True,
            mars=True,
            moon_mars_link=False,
        )
        lookahead_resolution = 15.0
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    actors = []
    data_providers = []
    message_location_tracker = MessageLocationTracker()

    transmission_actor = LinkTransmissionActor(
        default_bandwidth=25e6 // 8,
        buffer_if_link_busy=True,
        reroute_on_link_down=True,
        message_location_tracker=message_location_tracker
    )
    actors.append(transmission_actor)

    if traffic == "none":
        traffic_actor = None
    elif traffic == "point_to_point" or traffic == "point_to_point_eos":
        message_config = []
        sat_ids: list[int] = constellation.satellites.ids

        message_size = 1e6 * 8 if traffic == "point_to_point" else 10e6 * 8
        message_rate = 10.0 if traffic == "point_to_point" else 11.5
        for i in range(traffic_scale):
            # Randomly choose source and destination satellites using a fixed seed for reproducibility
            random.seed(i)
            source, destination = random.sample(sat_ids, 2)
            message_config.append((
                f"Traffic-{i}",
                source,
                destination,
                message_size,
                message_rate,
            ))
        traffic_actor = MultiPointToPointTrafficActor(
            message_config,
            update_interval=60,
            reliable_messages=reliable_messages,
            cutoff=6000,
        )
    elif traffic == "random":
        traffic_actor = RandomTrafficActor(
            satellites=constellation.satellites.ids,
            message_interval=1.0 / traffic_scale,
            message_size = NormalSampler(mean=1e6, std=1e6 // 4),
            message_source=UniformSampler(min=0, max=len(constellation.satellites.ids) - 1),
            message_destination=UniformSampler(min=0, max=len(constellation.satellites.ids) - 1),
            update_interval=60,
            reliable_messages=reliable_messages,
        )
    else:
        raise ValueError(f"Unknown traffic type: {traffic}")
    if traffic_actor is not None:
        actors.append(traffic_actor)

    custom_reference_loss_config = CustomScenarioLossConfig(seed=0, default_loss_probability=loss, max_frame_size=max_segment_size) if loss else None
    loss_config = LossConfig(seed=0, default_loss_probability=loss) if loss else None

    if delivery == "best_effort":
        routing_data_provider = BestEffortRoutingDataProvider()
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=False,
            model_bandwidth=True,
            loss_config=custom_reference_loss_config,
        )
    elif delivery == "store_and_forward":
        routing_data_provider = LookaheadRoutingDataProvider(
            resolution = lookahead_resolution,
            num_steps = 600,
        )
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=True,
            model_bandwidth=True,
            loss_config=custom_reference_loss_config,
        )
    elif delivery == "ltp":
        routing_data_provider = LookaheadRoutingDataProvider(
            resolution = lookahead_resolution,
            num_steps = 600,
        )
        routing_actor = MessageRoutingActor(
            routing_data_provider,
            store_and_forward=True,
            model_bandwidth=True,
            loss_config=loss_config,
            reliable_transfer_config=LTPConfig(max_segment_size=max_segment_size),
        )

        retransmission_config = RetransmissionConfig(max_retries=100)
        ltp_retransmission_actor = LTPMessageRetransmissionActor(config=retransmission_config,
                                                                 model_bandwidth=True,
                                                                 message_location_tracker=message_location_tracker)
        actors.append(ltp_retransmission_actor)
    else:
        raise ValueError(f"Unknown delivery method: {delivery}")

    actors.append(routing_actor)
    data_providers.append(routing_data_provider)

    if logging:
        preprocessed_logging_actor = PreprocessedLoggingActor(log_other=False)
        bw_logging_actor = BandwidthLoggingActor()
        ltp_logging_actor = LTPTransmissionLoggingActor()
        logging_actors = [preprocessed_logging_actor, bw_logging_actor, ltp_logging_actor]
    else:
        logging_actors = []

    simulation = Simulation(
        constellation,
        actors=actors,
        logging_actors= logging_actors + ([LoggingActor(verbose=True)] if verbose else []),
        data_providers=data_providers,
        timestep=15.0,
    )

    simulation.initialize(time=0)
    simulation_time = 6000 * 4 if logging else 6000
    simulation.run(simulation_time, progress=False)

    if logging:
        direct_messages = preprocessed_logging_actor.direct_messages
        broadcast_messages = preprocessed_logging_actor.broadcast_messages
        other_events = preprocessed_logging_actor.other_events

        default_bandwidth = transmission_actor._default_bandwidth

        period = 1.0
        output_file = os.path.join(results_dir, f"{scenario}-{delivery}-{loss}-{traffic}-{traffic_scale}-{walker_scale}.pickle")
        with open(output_file, "wb") as f:
            pickle.dump((direct_messages, broadcast_messages, other_events), f)

        bw_output_file = os.path.join(results_dir, f"{scenario}-{delivery}-{loss}-{traffic}-{traffic_scale}-{walker_scale}-bw-{period}.pickle")
        with open(bw_output_file, "wb") as f:
            pickle.dump((bw_logging_actor.aggregate(period=period, default_bandwidth=default_bandwidth)), f)

        ltp_output_file = os.path.join(results_dir, f"{scenario}-{delivery}-{loss}-{traffic}-{traffic_scale}-{walker_scale}-ltp-{period}.pickle")
        with open(ltp_output_file, "wb") as f:
            pickle.dump((ltp_logging_actor.aggregate(period=period)), f)

def get_parser():
    parser = argparse.ArgumentParser(description="Run a custom reference scenario in DSNS.")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["walker", "walker_large", "cubesat", "lunar_mars"], # Run: "walker", "cubesat", "lunar_mars"
        default="walker",
        help="The scenario to run"
    )
    parser.add_argument(
        "--walker-scale",
        type=int, # 1 - don't set
        default=1,
        help="Scale factor for the number of satellites in the walker and walker_large scenarios"
    )
    parser.add_argument(
        "--traffic",
        type=str,
        choices=["none", "point_to_point", "point_to_point_eos", "random"], # Run: point_to_point
        default="point_to_point",
        help="The type of traffic to simulate"
    )
    parser.add_argument(
        "--traffic-scale",
        type=int, # 10 - don't set
        default=10,
        help="Scale factor for traffic generation"
    )
    parser.add_argument(
        "--delivery",
        type=str,
        choices=["best_effort", "store_and_forward", "ltp"], # Run: "best_effort", "store_and_forward", "ltp"
        default="store_and_forward",
        help="The delivery method to use"
    )
    parser.add_argument(
        "--loss",
        type=float,
        default=None, # Run: 0.05
        help="Loss probability for best effort delivery"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="custom-references",
        help="Directory to store the results in"
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable logging of messages and bandwidth"
    )

    return parser

if __name__ == "__main__":
    parser = get_parser()

    args = parser.parse_args()

    main(
        scenario=args.scenario,
        walker_scale=args.walker_scale,
        traffic=args.traffic,
        traffic_scale=args.traffic_scale,
        delivery=args.delivery,
        loss=args.loss,
        verbose=args.verbose,
        results_dir=args.results_dir,
        logging=not args.no_logging,
    )

"""
parallel -j 9 ::: \
  "echo walker best_effort" \
  "python custom_reference.py --scenario walker --traffic point_to_point --delivery best_effort --loss 0.05 --results-dir custom-references" \
  "echo walker store_and_forward" \
  "python custom_reference.py --scenario walker --traffic point_to_point --delivery store_and_forward --loss 0.05 --results-dir custom-references" \
  "echo walker ltp" \
  "python custom_reference.py --scenario walker --traffic point_to_point --delivery ltp --loss 0.05 --results-dir custom-references" \
  "echo cubesat best_effort" \
  "python custom_reference.py --scenario cubesat --traffic point_to_point --delivery best_effort --loss 0.05 --results-dir custom-references" \
  "echo cubesat store_and_forward" \
  "python custom_reference.py --scenario cubesat --traffic point_to_point --delivery store_and_forward --loss 0.05 --results-dir custom-references" \
  "echo cubesat ltp" \
  "python custom_reference.py --scenario cubesat --traffic point_to_point --delivery ltp --loss 0.05 --results-dir custom-references" \
  "echo lunar_mars best_effort" \
  "python custom_reference.py --scenario lunar_mars --traffic point_to_point --delivery best_effort --loss 0.05 --results-dir custom-references" \
  "echo lunar_mars store_and_forward" \
  "python custom_reference.py --scenario lunar_mars --traffic point_to_point --delivery store_and_forward --loss 0.05 --results-dir custom-references" \
  "echo lunar_mars ltp" \
  "python custom_reference.py --scenario lunar_mars --traffic point_to_point --delivery ltp --loss 0.05 --results-dir custom-references"
"""