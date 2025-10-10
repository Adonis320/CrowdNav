from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Robot(Agent):
    def __init__(self, config, section, obstacles, bounds):
        super().__init__(config, section, obstacles, bounds)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob, obstacles=self.obstacles)
        action = self.policy.predict(state)
        return action

    def actWithIndex(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob, obstacles=self.obstacles)
        action, index = self.policy.predict(state, return_index=True)
        return action, index