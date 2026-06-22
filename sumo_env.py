"""SUMO environment for Module 10: DQN-based traffic signal control."""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import traci
except ImportError as exc:
    raise ImportError("Install SUMO Python tools: pip install traci sumolib") from exc


IN_EDGES = ["north_in", "south_in", "east_in", "west_in"]
NS_EDGES = ["north_in", "south_in"]
EW_EDGES = ["east_in", "west_in"]
TLS_ID = "center"


class SumoIntersectionEnv:
    """Small RL environment around a single SUMO traffic-light junction.

    State vector:
        [queue_n, queue_s, queue_e, queue_w,
         wait_n, wait_s, wait_e, wait_w,
         current_green_is_ns, phase_age]

    Actions:
        0 -> serve north-south
        1 -> serve east-west
    """

    def __init__(
        self,
        sumocfg: str = "sumocfg.xml",
        gui: bool = False,
        episode_seconds: int = 1800,
        min_green: int = 10,
        yellow_time: int = 3,
        seed: int | None = None,
        pcv: float = 1.0,
    ) -> None:
        self.sumocfg = sumocfg
        self.gui = gui
        self.episode_seconds = episode_seconds
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.seed = seed
        self.pcv = pcv
        self.step_count = 0
        self.current_action = 0
        self.phase_age = 0
        self.started = False

    @property
    def state_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        return 2

    def _sumo_binary(self) -> str:
        return "sumo-gui" if self.gui else "sumo"

    def _ensure_network(self) -> None:
        if not Path("intersection.net.xml").exists():
            subprocess.run(
                ["netconvert", "-n", "nodes.xml", "-e", "edges.xml", "-o", "intersection.net.xml"],
                check=True,
            )

    def _write_random_routes(self) -> None:
        rng = random.Random(self.seed)
        ns_rate = rng.randint(150, 900)
        ew_rate = rng.randint(150, 900)
        routes = f'''<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" guiShape="passenger" carFollowModel="IDM"/>
    <flow id="north_south" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ns_rate}" from="north_in" to="south_out"/>
    <flow id="south_north" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ns_rate}" from="south_in" to="north_out"/>
    <flow id="east_west" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ew_rate}" from="east_in" to="west_out"/>
    <flow id="west_east" type="car" begin="0" end="{self.episode_seconds}" vehsPerHour="{ew_rate}" from="west_in" to="east_out"/>
</routes>
'''
        Path("routes.xml").write_text(routes, encoding="utf-8")

    def reset(self) -> np.ndarray:
        self.close()
        self._ensure_network()
        self._write_random_routes()
        cmd = [self._sumo_binary(), "-c", self.sumocfg, "--no-warnings", "true"]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]
        traci.start(cmd)
        self.started = True
        self.step_count = 0
        self.current_action = 0
        self.phase_age = 0
        self._set_green(0)
        for _ in range(3):
            traci.simulationStep()
        return self._get_state()

    def close(self) -> None:
        if self.started:
            try:
                traci.close(False)
            except Exception:
                pass
            self.started = False

    def _set_green(self, action: int) -> None:
        # Most generated 4-way TLS programs use phase 0 for one direction pair and phase 2 for the other.
        # If a network produces fewer phases, fall back to phase 0.
        try:
            phases = traci.trafficlight.getCompleteRedYellowGreenDefinition(TLS_ID)[0].phases
            phase_index = 0 if action == 0 else min(2, len(phases) - 1)
            traci.trafficlight.setPhase(TLS_ID, phase_index)
        except Exception:
            pass
        self.current_action = action
        self.phase_age = 0

    def _yellow_transition(self) -> None:
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self.step_count += 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if action != self.current_action and self.phase_age >= self.min_green:
            self._yellow_transition()
            self._set_green(action)

        traci.simulationStep()
        self.step_count += 1
        self.phase_age += 1

        state = self._get_state()
        queues = self._queues()
        waits = self._waiting_times()
        total_queue = float(sum(queues.values()))
        total_wait = float(sum(waits.values()))
        reward = -(total_wait + 5.0 * total_queue)
        done = self.step_count >= self.episode_seconds or traci.simulation.getMinExpectedNumber() <= 0
        info = {"total_queue": total_queue, "total_wait": total_wait}
        return state, reward, done, info

    def _sample_visible(self, value: float) -> float:
        # Partial CV detection: pcv=1.0 sees all vehicles; lower pcv sees a fraction.
        return value * self.pcv

    def _queues(self) -> Dict[str, float]:
        return {edge: self._sample_visible(traci.edge.getLastStepHaltingNumber(edge)) for edge in IN_EDGES}

    def _waiting_times(self) -> Dict[str, float]:
        return {edge: self._sample_visible(traci.edge.getWaitingTime(edge)) for edge in IN_EDGES}

    def _get_state(self) -> np.ndarray:
        queues = self._queues()
        waits = self._waiting_times()
        q = [queues[e] / 30.0 for e in IN_EDGES]
        w = [waits[e] / 300.0 for e in IN_EDGES]
        phase = [1.0 if self.current_action == 0 else 0.0, min(self.phase_age / 60.0, 1.0)]
        return np.array(q + w + phase, dtype=np.float32)


def fixed_time_policy(env: SumoIntersectionEnv, cycle: int = 25) -> Dict[str, float]:
    state = env.reset()
    total_reward = 0.0
    infos: List[Dict[str, float]] = []
    done = False
    while not done:
        action = 0 if (env.step_count // cycle) % 2 == 0 else 1
        state, reward, done, info = env.step(action)
        total_reward += reward
        infos.append(info)
    env.close()
    return {
        "reward": total_reward,
        "avg_queue": float(np.mean([x["total_queue"] for x in infos])) if infos else 0.0,
        "avg_wait": float(np.mean([x["total_wait"] for x in infos])) if infos else 0.0,
    }
