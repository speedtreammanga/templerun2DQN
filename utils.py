import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

DEATH_SCREEN_PIXELS_COLORS = [4040, 4640, 4760]  # rgb values of column sum representing death screen
DEATH_SCREEN_PIXELS_COLORS_GRAYSCALE = 4600


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    n = len(scores)
    running_avg = np.empty(n)
    for t in range(n):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def plot_games_time(x, games_time, nb_games, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, games_time, color="C0")
    ax.set_xlabel("Games", color="C0")
    ax.set_ylabel("Game Time (Seconds)", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    plt.xticks(np.arange(1, nb_games, nb_games/20))
    plt.savefig(filename)


def process_frame(frame, gray_scale=False, downscale_factor=33.33):
    """
    Process images sequentially, downscales them.
    :param frame: single frame (3d rgb pixels).
    :param gray_scale: convert frame to grayscale or not.
    :param downscale_factor: downscaling factor in percentage.
    :return: returns same pictures but processed.
    """
    width = int(frame.shape[1] * (100 - downscale_factor)/100)
    height = int(frame.shape[0] * (100 - downscale_factor)/100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # TODO: return frame as torch Tensor
    return frame if gray_scale is False else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

"""
scaledown_factor: 33.33 => frame[65:85, 55]
scaledown_factor: 80% => frame[40:50, 40]
"""
def does_observation_contains_death_screen(observation, device, gray_scale):
    """
    Returns whether or not an observation contains a frame with the death screen.
    The death screen contains a big square with text about score, time, etc and has a specific background color.
    This method takes a column of rgb pixels from each frame and sums them up.
    If a frame contains the death screen then the rgb sum of the pixel column should add up to very specific values.
    :param observation: an array of rgb frames.
    :return: Bool: whether or not a frame contains the death screen.
    """

    for frame in observation:
        # pixels_column = frame[65:85, 55]  # grab a column of pixels from frame
        pixels_column = frame[40:60, 40]
        rgb_pixels_sum = np.sum(pixels_column, axis=0)  # sum every column in the RGB space

        if gray_scale and rgb_pixels_sum == DEATH_SCREEN_PIXELS_COLORS_GRAYSCALE:
            print("===== GAME OVER =====")
            return True

        if not gray_scale and np.array_equal(rgb_pixels_sum, DEATH_SCREEN_PIXELS_COLORS):  # assert for equality
            print("===== GAME OVER =====")
            return True

    return False


def sleep(seconds):
    for t in range(1, seconds+1):
        time.sleep(1)
        print(seconds+1 - t, "...")
