"""Implementation of a simple deterministic agent using Docker."""

from pommerman import agents
from pommerman.runner import DockerAgentRunner


class MyAgent(DockerAgentRunner):
    '''An example Docker agent class'''

    def __init__(self):
        self._agent = agents.RandomAgent()

    def act(self, observation, action_space):
        import random
        action_space = [ 0, 1, 2, 3, 4, 5 ]
        return random.choice(action_space)
        # return self._agent.act(observation, action_space)

def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent()
    agent.run()


if __name__ == "__main__":
    main()
