import collections
import time
import numpy as np
from gym import TempleRunGym
from dqn_agent import Agent
import utils


if __name__ == '__main__':
    utils.sleep(3)
    env = TempleRunGym(observation_size=4, downscale_img=80, gray_scale=True)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=200, eps_end=0.01, max_mem_size=20000,
                  input_dims=env.observation_shape, lr=0.003, n_actions=len(env.action_space))

    # agent.load('C:\\Users\\Alexandre\\Documents\\(C) Code\\ETS\\ELE767\\linear regression\\atari2\\models\\nbGames100_avgScore-0.37_trainingTime839_batchSize120_maxMemSize12000.pt')

    T_device = agent.Q_eval.device
    scores, eps_history = [], []
    n_games = 200

    games_time = []

    time1 = time.time()

    kk = False

    for i in range(n_games):
        time2 = time.time()
        score = 0
        done = False
        game_actions = []

        env.unpause_game()
        time.sleep(3)

        observation = env.reshape_frames_for_conv_layer(env.get_observation(device=T_device))

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done = env.step(action=action, device=T_device, game_time_start=time2)
            score += reward

            if done:
                agent.prune_transitions(2)  # get rid of death scene transitions, where 2 == approx...

            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            game_actions.append(env.action_space[action])

            if agent.mem_counter >= agent.batch_size and not kk:
                kk = True
                print("agent.mem_counter >= agent.batch_size")

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores)

        game_duration = time.time() - time2
        games_time.append(game_duration)

        print(collections.Counter(game_actions))
        print('game', i, ', score %.2f' % score,
              ', avg score %.2f' % avg_score,
              ', epsilon %.2f' % agent.epsilon,
              ', game duration %2f seconds' % game_duration,
              ', training time %.2f seconds' % (time.time() - time1))

    agent.save(nb_games=n_games, avg_score=avg_score, training_time=int(time.time() - time1))

    x = [i+1 for i in range(n_games)]
    utils.plot_learning_curve(x, scores, eps_history, 'templerun_2022_4.png')
    utils.plot_games_time(x, games_time, nb_games=n_games, filename='tp2_games_time_4.png')
