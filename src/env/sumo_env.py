from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np

try:
    import traci  # type: ignore
    from sumolib import checkBinary  # type: ignore
except Exception:  # pragma: no cover - allows coding without SUMO installed
    traci = None  # type: ignore
    checkBinary = None  # type: ignore


@dataclass
class VNWeights:
    motorcycle: float = 1.5
    car: float = 1.0
    bus: float = 2.0
    truck: float = 2.0


@dataclass
class EnvConfig:
    sumocfg_path: str
    tls_id: str
    phases: List[int]
    step_length: float = 1.0
    action_duration: int = 5  # seconds per chosen phase
    max_steps: int = 3600
    warmup_steps: int = 0
    vn_weights: VNWeights = field(default_factory=VNWeights)
    reward_type: str = "queue_delay"  # or "throughput"
    reward_speed_weight: float = 0.0  # bonus for higher speed
    reward_throughput_weight: float = 0.0  # bonus per vehicle passed
    gui: bool = False


class SumoMDPEnv:
    """SUMO-based MDP environment tailored for Vietnamese intersections.

    State (example): concatenation of per-lane features around a junction:
    - queue length
    - avg speed
    - waiting time
    - class-weighted counts (motorcycles prominent in VN)

    Action: select one of configured signal phases in `phases`.

    Reward: negative weighted queue and delay (or throughput), configurable.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self._ensure_sumo()
        self._tls = cfg.tls_id
        self._phases = cfg.phases
        self.action_space = len(cfg.phases)

        # Will be filled on first reset after connecting to SUMO
        self._lanes: List[str] = []
        self._state_dim = 0
        self._step = 0
        self._last_phase_index = 0
        self._connected = False
        self._arrived_prev = 0

    # ------------- SUMO management -------------
    def _ensure_sumo(self) -> None:
        if traci is None:
            raise RuntimeError(
                "traci/sumo not available. Install SUMO and ensure SUMO_HOME is set."
            )
        # Try to auto-detect pip-installed SUMO
        if "SUMO_HOME" not in os.environ:
            try:
                import sumo  # type: ignore[import-not-found]
                sumo_home = os.path.dirname(sumo.__file__)
                os.environ['SUMO_HOME'] = sumo_home
                print(f"✓ Auto-detected SUMO from pip: {sumo_home}")
            except ImportError:
                # Fallback for common macOS Homebrew install
                default_mac = "/opt/homebrew/opt/sumo/share/sumo"
                if os.path.isdir(default_mac):
                    os.environ['SUMO_HOME'] = default_mac
                    print(f"✓ Using default macOS SUMO_HOME: {default_mac}")
                else:
                    raise RuntimeError("SUMO_HOME is not set. export SUMO_HOME=... and try again.")

    def _start_sumo(self) -> None:
        if self._connected:
            return
        binary = checkBinary("sumo-gui" if self.cfg.gui else "sumo")  # type: ignore
        
        # Enhanced visualization settings for realistic 2D GUI
        gui_args = [
            binary, "-c", self.cfg.sumocfg_path,
            "--step-length", str(self.cfg.step_length),
        ]
        
        if self.cfg.gui:
            # GUI-specific settings for realistic 2D intersection visualization
            gui_args.extend([
                # Window configuration for good visibility
                "--window-size", "1400,900",
                "--window-pos", "50,50",
                # Slow down simulation for visibility (ms per simulation step)
                "--delay", "100",
            ])
        
        traci.start(gui_args)  # type: ignore
        self._connected = True

    def _close_sumo(self) -> None:
        if self._connected:
            traci.close(False)  # type: ignore
            self._connected = False

    # ------------- MDP API -------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if self._connected:
            self._close_sumo()
        self._start_sumo()

        # warmup
        for _ in range(self.cfg.warmup_steps):
            traci.simulationStep()  # type: ignore

        # collect lanes controlled by this TLS
        assert traci is not None, "traci not available"
        lanes = traci.trafficlight.getControlledLanes(self._tls)  # type: ignore
        self._lanes = sorted(list(set([l for l in lanes if l != ":"])))  # type: ignore

        self._state_dim = len(self._lanes) * 4  # 4 features per lane
        self._step = 0
        self._last_phase_index = 0
        self._arrived_prev = int(traci.simulation.getArrivedNumber())  # type: ignore

        # set initial phase
        traci.trafficlight.setPhase(self._tls, 0)
        return self._get_state()

    def step(self, action: int):
        assert traci is not None, "traci not available"
        action = int(np.clip(action, 0, self.action_space - 1))
        self._apply_phase(action)

        # advance simulation for action_duration seconds
        sim_steps = int(self.cfg.action_duration / self.cfg.step_length)
        start_arrived = int(traci.simulation.getArrivedNumber())  # type: ignore
        for _ in range(sim_steps):
            traci.simulationStep()  # type: ignore
            self._step += 1
        end_arrived = int(traci.simulation.getArrivedNumber())  # type: ignore
        vehicles_passed = max(0, end_arrived - start_arrived)
        self._arrived_prev = end_arrived

        s2 = self._get_state()
        # Aggregate metrics for logging and reward
        queue_total = 0.0
        wait_total = 0.0
        speeds: List[float] = []
        occupancies: List[float] = []
        halting_vehicles = 0
        for lane in self._lanes:
            queue_total += float(traci.lane.getLastStepHaltingNumber(lane))  # type: ignore
            halting_vehicles += int(traci.lane.getLastStepHaltingNumber(lane))  # type: ignore
            wait_total += float(traci.lane.getWaitingTime(lane))  # type: ignore
            speeds.append(float(traci.lane.getLastStepMeanSpeed(lane)))  # type: ignore
            try:
                occupancy = float(traci.lane.getLastStepOccupancy(lane))  # type: ignore
                occupancies.append(occupancy)
            except:
                pass
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        avg_occupancy = float(np.mean(occupancies)) if occupancies else 0.0

        r = self._compute_reward(
            queue_total=queue_total,
            wait_total=wait_total,
            avg_speed=avg_speed,
            vehicles_passed=vehicles_passed,
        )
        done = self._step >= self.cfg.max_steps or traci.simulation.getMinExpectedNumber() <= 0  # type: ignore
        info = {
            "step": self._step,
            "phase": action,
            "queue_length": queue_total,
            "avg_wait": wait_total / max(len(self._lanes), 1),
            "avg_speed": avg_speed,
            "vehicles_passed": vehicles_passed,
            "occupancy": avg_occupancy,
            "halting_vehicles": halting_vehicles,
        }
        return s2, r, done, info

    def close(self) -> None:
        self._close_sumo()

    # ------------- Helpers -------------
    def _apply_phase(self, idx: int) -> None:
        # Map action index to phase index in the TLS program
        assert traci is not None, "traci not available"
        try:
            traci.trafficlight.setPhase(self._tls, idx)  # type: ignore
            self._last_phase_index = idx
        except Exception:  # TraCIException
            # fallback to existing phase
            pass

    def _get_state(self) -> np.ndarray:
        assert traci is not None, "traci not available"
        feats: List[float] = []
        for lane in self._lanes:
            q_len = float(traci.lane.getLastStepHaltingNumber(lane))  # type: ignore
            avg_speed = float(traci.lane.getLastStepMeanSpeed(lane))  # type: ignore
            wait_time = float(traci.lane.getWaitingTime(lane))  # type: ignore

            # VN specific: emphasize motorcycle prevalence by weighted counts
            veh_ids = traci.lane.getLastStepVehicleIDs(lane)  # type: ignore
            weighted = 0.0
            for vid in veh_ids:
                vtype_id: str = str(traci.vehicle.getTypeID(vid)).lower()  # type: ignore
                # Match against generated Vietnamese vehicle types
                if "motorcycle" in vtype_id or "motor" in vtype_id or "moto" in vtype_id or "xe_may" in vtype_id:
                    weighted += self.cfg.vn_weights.motorcycle
                elif "bus" in vtype_id or "xe_buyt" in vtype_id:
                    weighted += self.cfg.vn_weights.bus
                elif "truck" in vtype_id or "xe_tai" in vtype_id:
                    weighted += self.cfg.vn_weights.truck
                else:
                    weighted += self.cfg.vn_weights.car

            feats.extend([q_len, avg_speed, wait_time, weighted])
        return np.array(feats, dtype=np.float32)

    def _compute_reward(
        self,
        queue_total: float,
        wait_total: float,
        avg_speed: float,
        vehicles_passed: int,
    ) -> float:
        """Compute reward with optional speed/throughput bonuses."""
        if self.cfg.reward_type == "throughput":
            return float(vehicles_passed + self.cfg.reward_speed_weight * avg_speed)

        # default: queue-delay penalty plus optional bonuses
        penalty = queue_total + (wait_total / 60.0)
        bonus = (self.cfg.reward_speed_weight * avg_speed) + (
            self.cfg.reward_throughput_weight * vehicles_passed
        )
        return float(-penalty + bonus)

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self.action_space
