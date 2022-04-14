# Deep Q-learning with Experience Replay
# Paper: https://arxiv.org/pdf/1312.5602v1.pdf

# inspiration: https://github.com/mswang12/minDQN/blob/main/minDQN.py

# import libraries
import numpy as np
import mss.tools
import keyboard
import utils

CAPTURE_BOX = {
    "top": 190,
    "left": 605,
    "width": 370,
    "height": 460
}
DISCOUNT_REWARD_FACTOR = 1
EXPERIENCE_REWARD_INDEX = 2
MINI_BATCH_SIZE = 5

with mss.mss() as sct:  # TODO: verify where to place this statement

    class TempleRunGym:
        def __init__(self, capture_box=CAPTURE_BOX, downscale_img=33.33, observation_size=4,
                     reward_game_over=-1, reward_still_alive=0.01, gray_scale=False):
            self.capture_box = capture_box
            self.gray_scale = gray_scale
            self.action_space_nothing_index = 0;
            self.action_space = ['UP', 'LEFT', 'DOWN', 'RIGHT']
            # self.action_space = ['NOTHING', 'UP', 'LEFT', 'DOWN', 'RIGHT']
            self.downscale_factor = downscale_img
            self.observation_size = observation_size
            self.reward_game_over = reward_game_over
            self.reward_still_alive = reward_still_alive
            conv_single_frame = self.reshape_frames_for_conv_layer([self.__get_one_frame('cpu')])  # reshape to match conv layers shape
            self.observation_shape = (observation_size, *conv_single_frame[0].shape)

        def __get_one_frame(self, device, show_frame=False):
            """
            Returns a 3d array of pixels: game frame (rgb only).
            :return: Array of rgb pixels.
            """
            img = sct.grab(self.capture_box)
            img = np.array(img)[:,:,:3]  # capture to np array and keep only (r,g,b) data, remove alpha channel
            img = utils.process_frame(img, downscale_factor=self.downscale_factor, gray_scale=self.gray_scale)
            return img

        def get_observation(self, device, show_frames=True):
            """
            Returns an array of frames (rgb pixels).
            :return: Array of images (rgb).
            """
            frames = []
            for i in range(self.observation_size):
                a_frame = self.__get_one_frame(device=device, show_frame=show_frames)
                frames.append(a_frame)

            return frames

        def unpause_game(self):
            self._act('SPACE')

        def reset(self):
            """
            Resets the state of the gym
            """
            pass

        def reshape_frames_for_conv_layer(self, frames):
            """
            Reshape a numpy array of frames to match a torch conv layer shape.
            :param frames: numpy array of frames.
            :return:
            """
            if self.gray_scale:
                for i, frame in enumerate(frames):
                    frames[i] = np.reshape(frame, (1, *frame.shape))
            else:
                for i, frame in enumerate(frames):
                    frames[i] = np.reshape(frame, (frame.shape[2], *frame.shape[:2]))

            return frames

        def _act(self, action):
            # if action != self.action_space[self.action_space_nothing_index]:
            keyboard.send(action)

        def step(self, action, game_time_start, device='cpu'):
            # print(self.action_space[action])
            self._act(self.action_space[action])
            observation = self.get_observation(device)
            done = utils.does_observation_contains_death_screen(observation, device, gray_scale=self.gray_scale)

            # if action == self.action_space_nothing_index and not done:
            #     reward = 0.015
            # elif not done:
            #     reward = self.reward_still_alive + ((time.time() - game_time_start) * 0.001)
            # elif done and action == self.action_space_nothing_index:
            #     reward = -0.5
            # else:
            #     reward = self.reward_game_over

            if done:
                reward = self.reward_game_over
            else:
                reward = 0.015

            # reward = self.reward_game_over if done else self.reward_still_alive

            reshaped_observation = self.reshape_frames_for_conv_layer(observation)

            return reshaped_observation, reward, done
