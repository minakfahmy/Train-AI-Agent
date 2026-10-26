#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.args = args

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        torch.set_num_threads(
            min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        )

        self.num_actions = env.action_space.n

        self.online = DQN(
            in_channels=4,
            num_actions=self.num_actions,
        ).to(self.device)

        self.target = DQN(
            in_channels=4,
            num_actions=self.num_actions,
        ).to(self.device)

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(),
            lr=args.learning_rate,
        )

        self.memory = deque(maxlen=args.buffer_size)
        self.recent_rewards = deque(maxlen=30)

        self.steps = 0
        self.episodes = 0

        self.checkpoint = Path(args.checkpoint)

        print(
            f"Device: {self.device}, actions: {self.num_actions}",
            flush=True,
        )

        # Resume training when --train_dqn_again is supplied.
        if args.train_dqn_again:
            saved = torch.load(
                self.checkpoint,
                map_location=self.device,
                weights_only=True,
            )

            self.online.load_state_dict(saved["model"])
            self.target.load_state_dict(self.online.state_dict())
            self.optimizer.load_state_dict(saved["optimizer"])

            self.steps = int(saved.get("steps", 0))
            self.episodes = int(saved.get("episodes", 0))
            self.recent_rewards.extend(saved.get("recent_rewards", []))

            print(
                f"Resumed {self.checkpoint} at step {self.steps}",
                flush=True,
            )
        
        
        
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            path = self.checkpoint

            if not path.exists():
                path = Path(__file__).resolve().parent / args.checkpoint

            if not path.exists():
                raise FileNotFoundError(
                    f"Cannot find checkpoint: {path}"
                )

            saved = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )

            self.online.load_state_dict(saved["model"])
            self.online.eval()

            print(f"Loaded checkpoint: {path}", flush=True)
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        if test:
            epsilon = self.args.test_epsilon
        else:
            fraction = min(
                1.0,
                self.steps / max(1, self.args.epsilon_decay),
            )

            epsilon = (
                self.args.epsilon_start
                + fraction
                * (
                    self.args.epsilon_end
                    - self.args.epsilon_start
                )
            )

        if random.random() < epsilon:
            action = random.randrange(self.num_actions)
        else:
            # Convert HWC pixels to CHW.
            pixels = np.ascontiguousarray(
                np.asarray(observation).transpose(2, 0, 1),
                dtype=np.uint8,
            )

            state = (
                torch.from_numpy(pixels)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                action = int(
                    self.online(state).argmax(dim=1).item()
                )
        return action
    
    def push(self, state, action, reward, next_state, terminated):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        ###########################
        state_pixels = np.ascontiguousarray(
            np.asarray(state).transpose(2, 0, 1),
            dtype=np.uint8,
        )

        next_pixels = np.ascontiguousarray(
            np.asarray(next_state).transpose(2, 0, 1),
            dtype=np.uint8,
        )

        self.memory.append(
            (
                state_pixels,
                int(action),
                float(reward),
                next_pixels,
                bool(terminated),
            )
        )
        
        
    def replay_buffer(self):
        batch = random.sample(
            self.memory,
            self.args.batch_size,
        )

        states, actions, rewards, next_states, terminals = zip(*batch)

        states = torch.as_tensor(
            np.stack(states),
            dtype=torch.uint8,
            device=self.device,
        )

        next_states = torch.as_tensor(
            np.stack(next_states),
            dtype=torch.uint8,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.int64,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        terminals = torch.as_tensor(
            terminals,
            dtype=torch.float32,
            device=self.device,
        )

        predicted_q = (
            self.online(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values

            target_q = (
                rewards
                + self.args.gamma
                * (1.0 - terminals)
                * next_q
            )

        loss = F.smooth_l1_loss(predicted_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return float(loss.item())
        
    def save(self):
        self.checkpoint.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = self.checkpoint.with_name(
            self.checkpoint.name + ".tmp"
        )

        torch.save(
            {
                "model": self.online.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps": self.steps,
                "episodes": self.episodes,
                "recent_rewards": list(self.recent_rewards),
            },
            temporary,
        )

        temporary.replace(self.checkpoint)

        print(
            f"SAVED {self.checkpoint} at step {self.steps}",
            flush=True,
        )

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################
        args = self.args

        if args.buffer_size < args.batch_size:
            raise ValueError(
                "buffer_size must be at least batch_size"
            )

        if min(
            args.batch_size,
            args.train_frequency,
            args.target_update,
            args.save_frequency,
        ) < 1:
            raise ValueError(
                "Batch size and update intervals must be positive"
            )

        self.online.train()

        self.env.seed(595)
        state = self.env.reset()

        episode_reward = 0.0
        episode_losses = []

        # Replay is refilled after resuming.
        warmup_until = self.steps + args.learning_starts

        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        append = (
            args.train_dqn_again
            and log_path.exists()
            and log_path.stat().st_size > 0
        )

        with log_path.open(
            "a" if append else "w",
            newline="",
        ) as stream:
            writer = csv.writer(stream)

            if not append:
                writer.writerow(
                    [
                        "episode",
                        "step",
                        "clipped_life_reward",
                        "mean30",
                        "mean_loss",
                    ]
                )

            try:
                while self.steps < args.total_steps:
                    action = self.make_action(
                        state,
                        test=False,
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = self.env.step(action)

                    # True termination stops bootstrapping.
                    # Time-limit truncation still permits it.
                    self.push(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated,
                    )

                    state = next_state
                    episode_reward += reward
                    self.steps += 1

                    if (
                        self.steps >= warmup_until
                        and len(self.memory) >= args.batch_size
                        and self.steps % args.train_frequency == 0
                    ):
                        loss = self.replay_buffer()
                        episode_losses.append(loss)

                    if self.steps % args.target_update == 0:
                        self.target.load_state_dict(
                            self.online.state_dict()
                        )

                    if self.steps % args.save_frequency == 0:
                        self.save()

                    if self.steps % 10_000 == 0:
                        print(
                            f"HEARTBEAT step={self.steps} "
                            f"replay={len(self.memory)}",
                            flush=True,
                        )

                    if terminated or truncated:
                        self.episodes += 1
                        self.recent_rewards.append(
                            float(episode_reward)
                        )

                        mean30 = float(
                            np.mean(self.recent_rewards)
                        )

                        mean_loss = (
                            float(np.mean(episode_losses))
                            if episode_losses
                            else float("nan")
                        )

                        writer.writerow(
                            [
                                self.episodes,
                                self.steps,
                                episode_reward,
                                mean30,
                                mean_loss,
                            ]
                        )

                        stream.flush()

                        print(
                            f"episode={self.episodes} "
                            f"step={self.steps} "
                            f"clipped_life_reward={episode_reward:.1f} "
                            f"mean30={mean30:.2f} "
                            f"loss={mean_loss:.5f}",
                            flush=True,
                        )

                        episode_reward = 0.0
                        episode_losses = []
                        state = self.env.reset()

            finally:
                self.save()
                self.env.close()
    
