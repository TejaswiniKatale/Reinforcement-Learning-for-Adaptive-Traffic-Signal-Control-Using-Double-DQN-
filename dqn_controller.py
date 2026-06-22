"""Module 10: DQN traffic signal controller for a SUMO 4-way intersection."""
from __future__ import annotations

import argparse
import csv
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sumo_env import SumoIntersectionEnv, fixed_time_policy

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000) -> None:
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.memory)


class DuelingDQN(nn.Module):
    """Small dueling DQN for the workshop state vector."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.value = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(64, 64), nn.ELU(), nn.Linear(64, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 1e-3,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer()

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q(s), dim=1).item())

    def learn(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN: online net chooses, target net evaluates.
            next_actions = torch.argmax(self.q(next_states), dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update()
        return float(loss.item())

    def soft_update(self) -> None:
        for target_param, local_param in zip(self.target.parameters(), self.q.parameters()):
            target_param.data.copy_((1.0 - self.tau) * target_param.data + self.tau * local_param.data)

    def save(self, path: str) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path: str) -> None:
        self.q.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.q.state_dict())


def train(args: argparse.Namespace) -> None:
    env = SumoIntersectionEnv(gui=args.gui, episode_seconds=args.seconds, pcv=args.pcv)
    agent = DQNAgent(env.state_dim, env.action_dim)
    eps_start, eps_end, eps_decay = 1.0, 0.01, max(args.episodes * 0.7, 1)
    Path("results").mkdir(exist_ok=True)

    with open("results/training_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "avg_queue", "avg_wait", "epsilon"])
        writer.writeheader()
        for ep in range(1, args.episodes + 1):
            env.seed = args.seed + ep
            state = env.reset()
            done = False
            total_reward = 0.0
            infos: List[dict] = []
            epsilon = eps_end + (eps_start - eps_end) * max(0.0, (eps_decay - ep) / eps_decay)

            while not done:
                action = agent.act(state, epsilon)
                next_state, reward, done, info = env.step(action)
                agent.memory.push(state, action, reward / 100.0, next_state, done)
                agent.learn()
                state = next_state
                total_reward += reward
                infos.append(info)

            row = {
                "episode": ep,
                "reward": round(total_reward, 3),
                "avg_queue": round(float(np.mean([x["total_queue"] for x in infos])), 3),
                "avg_wait": round(float(np.mean([x["total_wait"] for x in infos])), 3),
                "epsilon": round(epsilon, 4),
            }
            writer.writerow(row)
            print(row)
            if ep % args.save_every == 0:
                agent.save(args.model)
    env.close()
    agent.save(args.model)
    print(f"Saved model to {args.model}")


def evaluate(args: argparse.Namespace) -> None:
    env = SumoIntersectionEnv(gui=args.gui, episode_seconds=args.seconds, pcv=args.pcv, seed=args.seed)
    agent = DQNAgent(env.state_dim, env.action_dim)
    agent.load(args.model)
    state = env.reset()
    total_reward = 0.0
    infos: List[dict] = []
    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        state, reward, done, info = env.step(action)
        total_reward += reward
        infos.append(info)
    env.close()
    print("DQN evaluation")
    print({
        "reward": round(total_reward, 3),
        "avg_queue": round(float(np.mean([x["total_queue"] for x in infos])), 3),
        "avg_wait": round(float(np.mean([x["total_wait"] for x in infos])), 3),
    })
    print("Fixed-time baseline")
    print(fixed_time_policy(SumoIntersectionEnv(gui=False, episode_seconds=args.seconds, pcv=args.pcv, seed=args.seed)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--pcv", type=float, default=1.0, help="connected vehicle visibility fraction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="dqn_model.pth")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()
    if args.train:
        train(args)
    elif args.eval:
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
