#solving the cartpole problem with q learning and a binnnig approach.
#the current observation gives us 4 variables which summarize the information about the state.
# we need to divide each of these variable into bins. say for example 10 bins
#each state would be a combination of the bin values of these 4 variables.
#so 10^4 states. Each state would have associated an appropriate action which would be learned by Q-Learning

import numpy as np
import gym
import matplotlib.pyplot as plt

def buildStates(bins):
	return int("".join(map(lambda s:str(int(s)), bins)))

def getBin(x, bins):
	return np.digitize(x=[x], bins=bins)[0]
	# since it takes both inputs as vectors

class FeatureTransformer:

	def __init__(self):
		self.cartPosition = np.linspace(-2.4,2,4,9)
		self.cartVelocity = np.linspace(-2,2,9)
		self.poleAngle = np.linspace(-0.4,0.4,9)
		self.poleVelocity = np.linspace(-3.5,3.5,9)

	def transform(self, observation):
		cartPosition, cartVelocity, poleAngle, poleVelocity = observation
		return buildStates([
			getBin(cartPosition, self.cartPosition),
			getBin(cartVelocity, self.cartVelocity),
			getBin(poleAngle, self.poleAngle),
			getBin(poleVelocity, self.poleVelocity)])

class Model:

	def __init__(self, featureTransformer, env):
		self.featureTransformer = featureTransformer
		self.env = env
		noOfStates = 10 ** env.observation_space.shape[0]
		noOfActions = env.action_space.n
		self.Q = np.random.uniform(low=-1, high=1, size=(noOfStates, noOfActions))

	def predict(self, observation):
		return self.Q[self.featureTransformer.transform(observation)]

	def getAction(self, observation, eps):
		if np.random.random() < eps:
			return self.env.action_space.sample()
		else:
			actions = self.predict(observation)
			return np.argmax(actions)

	def updateQ(self, observation, action, g):
		state = self.featureTransformer.transform(observation)
		self.Q[state][action] += 0.01 * (g - self.Q[state][action])




def playGame(model, eps, gamma):
	observation = env.reset()
	done = False
	totalReward = 0
	t = 0
	while not done and t < 10000:
		# with the new gym version, the total number of iteration that one can go is retricted to 200
		#env.render()
		t += 1

		action = model.getAction(observation, eps)
		oldObservation = observation
		observation, reward, done, info = env.step(action)
		totalReward += reward
		if done and t < 199:
			reward = -300
		g = reward + gamma*np.max(model.predict(observation))
		model.updateQ(oldObservation, action, g)
	return totalReward

def runningAverage(totalRewards):
	n = len(totalRewards)
	runningAverage = np.empty(n)
	for i in range(n):
		runningAverage[i] = totalRewards[max(0,i-100):i+1].mean()
	plt.plot(runningAverage)
	plt.show()

if __name__ == '__main__':
	
	env = gym.make('CartPole-v0')
	featureTransformer = FeatureTransformer()
	model = Model(featureTransformer, env)
	gamma = 0.9

	'''if 'monitor' in sys.argv:
	filename = os.path.basename(__file__).split('.')[0]
	monitor_dir = './' + filename + '_' + str(datetime.now())
	env = wrappers.Monitor(env, monitor_dir)'''

	N = 10000
	totalRewards = np.empty(N)
	for n in range(N):
		eps = 1.0/np.sqrt(n+1)
		totalReward = playGame(model, eps, gamma)
		totalRewards[n] = totalReward
		if n % 100 == 0:
			print("episode:", n, "total reward:", totalReward, "eps:", eps)
	print("avg reward for last 100 episodes:", totalRewards[-100:].mean())
	print("total steps:", totalRewards.sum())

	plt.plot(totalRewards)
	plt.title("Rewards")
	plt.show()

	runningAverage(totalRewards)

