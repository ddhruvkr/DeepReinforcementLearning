import gym
import numpy as np
import matplotlib.pyplot as plt
# right now it is just using a random strategy of selection one out of the 2 decisions, either to go left or right

def getAction(obs, W):
	return 1 if np.dot(obs,W) > 0 else 0

def playEpisode(env, W):
	observation = env.reset()
	done = False
	t = 0
	while not done and t <= 1000:
		# with the new gym version, the total number of iteration that one can go is retricted to 200
		#env.render()
		t += 1
		action = getAction(observation, W)
		observation, reward, done, info = env.step(action)
	return t

def playEpisodes(env, W):
	length = 0
	for t in range(10):
		length += playEpisode(env, W)
	avgLength = length/10
	print(avgLength)
	return avgLength


def randomSearch(env):
	episodeLengths = []
	best = 0
	optimumW = None
	for t in range(100):
		W = np.random.random(4)*2 - 1
		avgEpisodeLength = playEpisodes(env, W)
		episodeLengths.append(avgEpisodeLength)
		if (best < avgEpisodeLength):
			best = avgEpisodeLength
			optimumW = W

	return optimumW, episodeLengths

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	W, episodeLengths = randomSearch(env)
	plt.plot(episodeLengths)
	plt.show()

	playEpisodes(env, W)