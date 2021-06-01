
> 4 is down, 3 is up
3 is down, 2 is up



Research ideas:
- disable MCTS, should approximate DQN maybe with some minor changes
- more generally, try to improve upon the "improved policy = visit counts" approach - especially for low rollout regime
- use GAE instead of vanilla discounting? does that even make sense?
- add autoencoder - maybe just early in training to help initialize hidden with useful stuff? maybe even just as pretraining
- i notice the dynamics network has trouble learning. i suspect that's because the instantaneous reward is hard to predict (sparse + no room for error), and the value is so smooth over the 5 training steps that we can just predict the same value for each (don't need a dynamics network) and do well enough. I wonder if we had an additional term which was basically a faster-decaying value that would help bootstrapping the learning.
- do DQN-style approach where we predict q value for each action in the same forward pass. would still be combined with dynamics network taking one action. basically equivalent to the policy network - except not normalized to a sum of 1. then we could train it to just match the MCTS value estimates? still a question of how to sample from this though.
    - mellowmax?
- instead of boostrapping with the on-policy MCTS improved value estimate, do this with a more off-policy approach like a DQN-style argmax, or even do a MCTS on the training side to get an on-policy improved value estimate.
- look into some of the off-policy correction schemes and apply?
- have two value functions, one short-term (high discounting) and one long-term (low discounting). the short-term one will be mainly for local stuff - basically avoiding death. the long-term one will be for higher-level strategy, like what rooms to visit in montezuma. is this sort of like hierarchical learning? short term one will probably be easier to learn and plan with, as it will change notably only after a few steps of MCTS search. the trick is how to figure out how to make the long term one useful in shallow planning searches.
- any easy ways to add hierarchical methods to this paper?
    - train another dynamics model where each step represents 20 frames. use the long-term value function mentioned above in this one, and the short term one in the other. use the discovered hidden state in the long-term planner as the "goal" input for the short-term planner. what is an "action" in the long-term planner? maybe just some random embedding.. not sure. could also be an input to the short-term model tho.

- it's unclear to me if the representation and dynamics models actually learn the same hidden representation, or if the heads just learn to be able to work with either.
    - *idea*: train output of dynamics model to be close to output of representation model for t+1 input. this essentially directly trains the forward-dynamics, but in the hidden dimension.

- MCTS I believe is unable to determine that two actions are identical (eg they are both no-ops), and will guide the policy towards giving them equivalent value. at larger scale this could ruin the ablitity to search deeply? should try to think about ways to avoid this.

- i'm really interested in having the model predict its own uncertainty, eg of value function. then if the model is highly confident of a prediction, we don't need to do MCTS (for example) on that action - and we can focus our search time on less certain areas of the tree


Architecture of the game player:
- Goals: 
    - must achieve within 2x of near-ideal throughput of GPU
    - thus, cannot bottleneck at CPU or elsewhere
    - and must keep GPU fed optimally
- Approach
    - vecevn
        - i was worried about stragglers - ie let's say that env.reset() takes 10x the time of env.step(), so every time an environment has to be reset it holds the whole group back. but that's only true if we're operating on multiple independent cores - here everything is actually sequentially executed on the CPU so this is invalid. nice!


The data that goes into the replay buffer:
Options:
- A full game, of variable length, but length ~1000 transitions.
    - issues: games can be extremely long (27000 transitions). since we play games at near-realtime, this is like 30 mins of eval time. in this time we will have updated the most recent model checkpoint a lot of times, and can't be waiting for data this long.
- Slices of a game, of length ~200 steps
    - how they did it in the paper
    - somewhat annoying to reason about. have to unnecessarily pad some observations, or else append obs from before the slice to the container.
- Individual training examples
    - contains k=5 steps, the full stacked observation, etc
    - very inefficient, ~32x less data efficient than the previous options. we're pushing towards memory limits in the replay buffer, so this isn't great
    - makes training code very clear
- Same as last one, but take additional steps to deal with memory inefficiency
    - store images by reference, with some sort of ref-counting garbage collector.
        - if we could have python's GC manage this for us, would be pretty slick actually
            - unfortunately it seems like ray doesn't handle shared obj references, unless we fall back to pickle (slow?)
    - compress observations
    - get high-mem instance for replay buffer, or shard replay buffer ie in ray object store



Each transition has:
- an action (TODO: taken before or after the observation?)
    - the training needs observation->action
- an observation
- immediate reward
- MCTS improved value
- MCTS improved policy
- TD error?


The training process:
- obtain a sequence of k=5 transitions taken according to a recent policy
    - plus num_framestack previous observations
- evaluate the representation model on the observation at the initial transition
    - compute loss for predictions at i=0
- for i in k=5:
    - step the hidden state with the dynamics model
    - compute loss for predictions at step=i
    - TODO: figure out if there's off-by-one error here



