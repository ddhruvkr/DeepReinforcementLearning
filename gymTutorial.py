import gym
env = gym.make('CartPole-v0')
env.reset()
box = env.observation_space
print(box)

print(env.action_space.sample())
observation, reward, done, info = env.step(0)

#running through an episode
done = False
while not done:
	observation, reward, done, info = env.step(env.action_space.sample())
	print(observation)
	print(reward)
	print(done)
	print(info)