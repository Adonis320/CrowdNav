from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    def __init__(self, config, section, obstacles, bounds):
        super().__init__(config, section, obstacles, bounds)

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob, obstacles=self.obstacles)
        action = self.policy.predict(state)
        return action
