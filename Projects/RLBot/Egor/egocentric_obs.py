from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils import common_values, math
import numpy as np
import math as py_math

class EgocentricObs(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.POS_STD = 1/2300.0
        self.ANG_STD = 1/py_math.pi
        # print('EgocentricObs finished __init__()')

    def reset(self, initial_state):
        pass

    def build_obs(self, player, state, prev_action) -> np.ndarray:
        # print('Building obs')
        if prev_action is None:
            print("!ATTEMPTED TO BUILD EGOCENTRIC OBS WITH NO PREV ACTIONS ARGUMENT!")
            raise AssertionError

        players = state.players
        # print('Hello, made it here')
        if player.team_num == common_values.ORANGE_TEAM:
            player_car = player.inverted_car_data
            ball = state.inverted_ball
        else:
            player_car = player.car_data
            ball = state.ball

        ob = []
        ob.append(prev_action)
        ob.append([int(player.has_flip),
                   player.boost_amount/100.0, #Check to ensure range is 0-100
                   int(player.on_ground)])

        ob.append(player_car.position * self.POS_STD) # The maps position relative to the player (kinda)
        ob.append(player_car.linear_velocity * self.POS_STD)
        ob.append(player_car.euler_angles() * self.ANG_STD)
        ob.append(player_car.angular_velocity * self.ANG_STD)

        ob.append(math.get_dist(ball.linear_velocity, player_car.linear_velocity) * self.POS_STD)
        ob.append(math.get_dist(ball.angular_velocity, player_car.angular_velocity) * self.ANG_STD)
        bp_rel_pos = math.get_dist(ball.position, player_car.position)
        pb_dist = math.vecmag(bp_rel_pos) * self.POS_STD
        ob.append([pb_dist])
        ob.append(math.unitvec(bp_rel_pos))

        # Since we invert the car and ball data when the agent is in the Orange team
        # the "Orange Goal" is always the enemy goal
        gp_rel_pos = math.get_dist(player_car.position, common_values.ORANGE_GOAL_CENTER)
        gp_dist = math.vecmag(gp_rel_pos) * self.POS_STD
        ob.append([gp_dist])
        ob.append(math.unitvec(gp_rel_pos))
        ogp_rel_pos = math.get_dist(player_car.position, common_values.BLUE_GOAL_CENTER)
        pog_dist = math.vecmag(ogp_rel_pos) * self.POS_STD
        ob.append([pog_dist])
        ob.append(math.unitvec(ogp_rel_pos))

        for other in players:
            if other.car_id == player.car_id:
                continue

            if player.team_num == common_values.ORANGE_TEAM:
                car_data = other.inverted_car_data
            else:
                car_data = other.car_data


            ob.append(math.get_dist(car_data.linear_velocity, player_car.linear_velocity) * self.POS_STD)
            ob.append(math.get_dist(car_data.angular_velocity, player_car.angular_velocity) * self.ANG_STD)
            op_rel_pos = math.get_dist(car_data.position, player_car.position)
            op_dist = math.vecmag(op_rel_pos) * self.POS_STD
            ob.append([op_dist])
            ob.append(math.unitvec(op_rel_pos))
            # NOT EGOCENTRIC, I feel like angle to world is more important, due to gravity effects
            ob.append(car_data.euler_angles() * self.ANG_STD)

        return np.concatenate(ob)
        