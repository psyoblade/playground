"""
A Work-In-Progress agent using Tensorforce
"""
from . import BaseAgent
from .. import characters
import os
import numpy as np

class TensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo', checkpoint='models/checkpoint'):
        super(TensorForceAgent, self).__init__(character)
        self.algorithm = algorithm
        self.checkpoint = checkpoint
        self.agent = None
        self.state = {}
        self.env = None

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        agent_state = self.env.featurize(obs)
        action = self.agent.act(agent_state)
        return action 

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import PPOAgent
        self.env = env

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)

            self.agent = PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=[
                    dict(type='dense', size=64),
                    dict(type='dense', size=64)
                ],
                batching_capacity=1000,
                step_optimizer=dict(type='adam', learning_rate=1e-4))
            
            self.restore_model_if_exists(self.checkpoint)

        return self.agent

    def restore_model_if_exists(self, checkpoint):
        if os.path.isfile(checkpoint):
            pardir = os.path.abspath(os.path.join(checkpoint, os.pardir))
            self.agent.restore_model(pardir)
            print("tensorforce model '{}' restored.".format(pardir))

    def save_model(self, checkpoint):
        pardir = os.path.abspath(os.path.join(checkpoint, os.pardir))
        if not os.path.exists(pardir):
            os.mkdir(pardir)
            print("checkpoint dir '{}' created.".format(pardir))
        checkpoint_path = agent.save_model(pardir, False)
        print("checkpoint model '{}' saved.".format(checkpoint_path))
