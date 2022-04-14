# Observations

---

## Entry 1
- Character learns to jump a lot.
- Also, need to add a `DO_NOTHING` action

##### Current reward strategy:
- +0.01 when action selected and not game_over
- -1 when gamer_over

##### Remarks:
- Too simplistic ?

---

## Entry 2
- Updated reward strategy and `DO_NOTHING` action
- Character learns to die: because doing nothing yields more points

##### Current reward strategy:
- +0.2 *(way too high)* when not game_over and action == `DO_NOTHING`
- +0.01 when not game_over
- -1 when game_over

##### Remarks:
- agent optimizes for reward policy instead of action ?
- Need a more complex reward strategy: like the brain...

---

## Entry 3
- Still learns to `DO_NOTHING` and die
- Will add a reward factor based on `game_time_start` - `env.step.time` for other actions
- Will probably need to change the model architecture