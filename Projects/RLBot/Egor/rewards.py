from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.gamestates import GameState, PlayerData, PhysicsObject
from rlgym.utils.common_values import ORANGE_TEAM, BLUE_TEAM, BACK_WALL_Y, BLUE_GOAL_CENTER, CAR_MAX_SPEED, ORANGE_GOAL_BACK, BLUE_GOAL_BACK, BALL_MAX_SPEED, BALL_RADIUS, CEILING_Z
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import ConditionalRewardFunction
import numpy as np

# REQUIRES PLAYER_TEAM = BLUE (or inverted)
def player_is_goalside(player_car_data: PhysicsObject, ball:PhysicsObject) -> bool: 
    return np.linalg.norm(player_car_data.position - BLUE_GOAL_CENTER) * 1.25 < np.linalg.norm(ball.position - BLUE_GOAL_CENTER)
            
def possession(player: PhysicsObject, opp: PhysicsObject, ball:PhysicsObject) -> int: #(-1 opponent possession, 0 niether, 1 player possession)
    player_dist = np.linalg.norm(ball.position - player.position)
    opp_dist = np.linalg.norm(ball.position - opp.position)

    if opp_dist * 2 < player_dist:
        return -1
    elif player_dist * 2 < opp_dist:
        return 1
    else:
        return 0

class PossessionReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Compensate for inside of ball being unreachable (keep max reward at 1)
        opp = None
        for other in state.players:
            if not other.car_id == player.car_id:
                opp = other
        return float(possession(player.car_data, other.car_data, state.ball))

class RewardIfGrounded(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player.on_ground

class RewardIfGoalside(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        return player_is_goalside(player.car_data, state.ball) 

class RewardIfShouldShadow1s(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        assert len(state.players) == 2, "RewardIfShouldShadow1s() only compatable with 1v1s"

        if player.team_num == BLUE_TEAM:
            ball = state.ball
            player_car_data = player.car_data
            inverted = False
        else:
            ball = state.inverted_ball
            player_car_data = player.inverted_car_data
            inverted = True

        ball_in_opp_third = ball.position[1] > (BACK_WALL_Y * 2/3)

        player_wider_than_ball = (abs(ball.position[0]) < abs(player_car_data.position[0])) and (ball.position[0] * player_car_data.position[0] > 0)

        opp_car_data = None
        for other in state.players:
            if not other.car_id == player.car_id:
                opp_car_data = other.car_data
                if inverted:
                    opp_car_data = other.inverted_car_data

        opp_in_possession = possession(player_car_data, opp_car_data, ball) == -1

        player_goalside = player_is_goalside(player_car_data, ball)

        # player is driving into own half (-ve y) and towards x=0
        player_retreating_towards_net = player_car_data.linear_velocity[1] < 0.0 and player_car_data.position[0] * player_car_data.linear_velocity[0] < 0

        return (not ball_in_opp_third) and player_wider_than_ball and opp_in_possession and player_goalside and player_retreating_towards_net

class RewardIfPlayerBallY(ConditionalRewardFunction):
    def condition(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> bool:
        xy_dist = np.linalg.norm(state.ball.position[:2] - player.car_data.position[:2])
        return xy_dist < 1000.0 and state.ball.position[1] > BALL_RADIUS+10.0

class PlayerBallYDistReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.thresh = BALL_RADIUS + 30

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ydist = state.ball.position[1] - player.car_data.position[1]
        rel_dist = (ydist - self.thresh)/CEILING_Z
        if rel_dist > 0:
            return 1.0 - rel_dist
        else: 
            return 1.0


class TimestepReward(RewardFunction):
    def __init__(self, factor=1000):
        super().__init__()
        self.steps = 0
        self.factor = factor

    def reset(self, initial_state: GameState):
        self.steps = 0
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self.steps = self.steps + 1
        multiplier = self.factor / (self.steps + self.factor) 
        print(f'multiplier: {multiplier}')
        return multiplier

    
class MultiplyRewards(RewardFunction):
    def __init__(self, reward_func1:RewardFunction, reward_func2:RewardFunction):
        super().__init__()
        self.reward_func1 = reward_func1
        self.reward_func2 = reward_func2

    def reset(self, initial_state: GameState):
        self.reward_func1.reset(initial_state)
        self.reward_func2.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward_func1.get_reward(player, state, previous_action) * self.reward_func2.get_reward(player, state, previous_action)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward_func1.get_final_reward(player, state, previous_action) * self.reward_func2.get_final_reward(player, state, previous_action)