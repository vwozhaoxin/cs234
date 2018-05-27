### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
	"""Evaluate the value function from a given policy.
# =============================================================================
# 
# =============================================================================
	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	#print(policy.shape)
	#print(nS,nA)
	if len(policy.shape) !=1:
		policy.reshape(-1,1)
	value = np.zeros(nS)
	#print(value)
	iterations = max_iteration
	#print(type(iterations))
	while iterations!=0 :
		iterations-=1
		value_old = value.copy()
		for i in reversed(range(nS)):
			#action = policy[i]
			#print(action)
			#print(P[i][action])
			#print(len(P[i][action]))
			#state = list(np.arange(nS))
			'''result = P[i][policy[i]]
			value[i] = np.array(result)[:,2].mean()
			for num in range(len(result)):
				probability,nextstate,reward,terminal = result[num]
				value[i] += probability*gamma *value_old[nextstate]'''
			p,s,r,t = P[i][policy[i]][0]  #  查看实际情况发现每次行动很自由一个节点 因此这样写简单
			value[i] = r+p*gamma*value_old[s]

			#states =[]
			#pros=[]
			#state =i
			#while True:
			#	a= policy[state]
		#		probability1,nextstate,reward1,_ =P[state][a][0]
		#		states.append(value_old[nextstate])
		#		pros.append(probability1*gamma) 
		#		if nextstate == state:
		#			break
		#		state = nextstate
		#	probability = np.cumprod(np.array(pros))
			#print(probability)
		#	value[i] = r + np.sum(probability*np.array(states))*p*gamma
		
			#tmp1 =0
			#tmp2 =0
			#for j in range(nA):
			#	p,n,r,_=P[i][j][0]
			#	tmp1 += p*value_old[n]
			#	tmp2 += r*p
 			#print(reward,iterations)
			#value[i]=tmp2+ gamma*tmp1

		if np.linalg.norm(value - value_old,ord=1)<tol:
			break
			#for j in range(nA):
		#		 P[i][j]
	#print(i)
	############################
	return value


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	############################
	# YOUR IMPLEMENTATION HERE #
	#print(policy.shape)
	if len(policy.shape) !=1:
		policy.reshape(-1,1)
	#q_value=0
	#new_policy = np.zeros(nS)
	q_value = np.zeros((nS,nA))
	for i in range(nS):
		#action = policy[i]
		#probability,nextstate,reward,_ =P[i][action][0]
		#result = P[i]
		for j in range(nA):
			p,n,r,_ =P[i][j][0]
			q_value[i,j]=p*value_from_policy[n]*gamma+r
	policy =np.argmax(q_value,axis=1)
	#print(policy.shape)
	new_policy=policy.astype(int)
	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	i=0
	policy = np.random.randint(nA,size=nS)
	#new_policy = np.zeros(nS)
	while  True:
		old_policy = policy.copy()
		#print(max_iteration)
		V = policy_evaluation(P, nS, nA, old_policy, gamma=gamma, max_iteration=max_iteration, tol=tol)
		i+=1
		policy = policy_improvement(P, nS, nA, V, old_policy, gamma=gamma)
		#print(policy,policy.shape)
		#print(V,policy,old_policy)
		if np.linalg.norm((policy - old_policy),ord=1) ==0:
			break
	#print(i)
	############################
	return V, policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
	V = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	k=1
	while k <max_iteration:
		value = V.copy()
		k+=1
		for state in range(nS):
			value_function = np.zeros(nA)
			for action in range(nA):
				results = P[state][action]
				value_function[action] = np.array(results)[:,2].mean()
				for result in results:
					p,s,r,t=result 
					value_function[action] += gamma*p *value[s]
			V[state]= np.amax(value_function) 
		
			policy[state] = np.argmax(value_function)			
		if np.linalg.norm(V-value,ord=1) <tol:
			break 	

	print(policy)


	############################
	return V, policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		print(a)
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(1000):
		env.render()
		time.sleep(0.1) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, terminal = env.step(a)
		#print(terminal)
		episode_reward += rew
		#print(done)
		if done:
			if episode_reward!=0:
				break
			else:
				env.reset()
		#if terminal:
		#	env.reset()
	assert done	
	env.render();
	print ("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
	env = gym.make("Stochastic-4x4-FrozenLake-v0")
	print (env.__doc__)
	#print ("Here is an example of state, action, reward, and next state")
	#example(env)
	start1= time.time()
	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=1000, tol=1e-3)
	render_single(env,p_vi)
	end1 = time.time()
	start2 = time.time()
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=1000, tol=1e-3)
	#print(env.P)
	render_single(env,p_pi)
	end2= time.time()
	print("vi cost time{}".format(end1-start1))
	print('pi cost time{}'.format(end2- start2))


	# 7.084802865982056  8*8 for pi