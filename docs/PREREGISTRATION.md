# Pre-registration

Written and committed **before** any grid training run or evaluation exists
(see git history). The simulator and trainer were only smoke-tested with
scripted policies and short throw-away runs when this was written. Anything in
the final report that is not listed here is labelled *exploratory*.

## Design

* Cells: 4 planets (`corrosion`, `jam`, `quake`, `combined`) x 3 missions
  (`search`, `haul`, `escort`) = 12.
* Arms per cell (one-factor-at-a-time around a default):

  | arm | comm | coupling | protect | answers |
  |---|---|---|---|---|
  | `default` | gossip | on | off | reference |
  | `silent` | silent | on | off | coordination |
  | `blackboard` | blackboard | on | off | coordination |
  | `solo` | gossip | off | off | tight coupling |
  | `protected` | gossip | on | on | sacrifice |

* Every arm is a separately trained shared-parameter PPO policy co-trained with
  a learned hazard adversary (half of the training envs adversarial, half
  "nature" random-walk hazards), swarm size and hazard tempo randomised per env.
  2 seeds per arm (more if compute allows; the number actually used is reported).
* Evaluation conditions: tempo in {0.1, 0.3, 0.6} x N in {2,3,4,6,8,12} x hazard
  mode in {nature, adversarial}. Adversarial score of a policy = the **minimum
  over a pool of learned adversaries from the same cell** of its mean `J`
  (pool = the adversary co-trained with that policy plus adversaries co-trained
  with other arms/seeds of the cell). Nature score = mean `J` under random-walk
  hazards. Policies act stochastically as trained.
* A *condition* for the principle analysis is (planet, mission, tempo); outcomes
  are averaged over N in {4,6,8} and over seeds unless stated.

## Outcomes

With `rel(a,b) = (a-b) / (0.5*(a+b) + 0.02)`:

* `V_comm = rel(J[blackboard], J[silent])`, `V_gossip = rel(J[default], J[silent])`
* `V_couple = rel(J[default], J[solo])`
* `V_sac = rel(J[default], J[protected])` (price of safety), and the loss rate
  (deaths per robot) of `default` vs `protected`
* size: `J(N)` of `default`; scaling exponent `beta` = slope of `log J` on
  `log N`; `N*(lambda) = argmax_N J(N) - lambda*N*J(2)/2` for `lambda` in {0.25, 0.5, 0.75}

## Principle variables (measured from `default`-arm rollouts, never from the outcome arms' differences)

* `S = tau_act / tau_stale` — `tau_stale`: mean first-passage time of a blob
  centre to a displacement > `SIGMA` (capped at T); `tau_act = 0.5 / mean
  realised speed of live robots`.
* `W(t)` for removal steps t in {0, 50, 100}: `(J_full - J_removed) * N / J_full`
  at N = 6. `R_late = W(100)`, `R_early = W(0)`.

## Primary test — cross-world transfer (the claim of the proposal)

For each outcome `y` in {`V_comm`, `V_gossip`, `V_couple`, `V_sac`, `beta`,
default-arm loss rate}, under the adversarial score:

* **Principle model P**: `y ~ mission intercepts + a*log(S) + b*R_late`,
  slopes shared across planets and missions (5 parameters), ordinary least squares.
* Baselines: **B0** grand mean; **B1** mission means; **B2** mission means +
  shared slope on raw `tempo`.
* Headline split: fit on the three single-hazard planets (27 conditions),
  predict the `combined` planet (9 conditions). Robustness: leave-one-planet-out
  over all four planets (36 predictions).
* Metrics: MAE, Spearman rho between predicted and observed, sign agreement.

The claim "a small number of shared principles lets one hostile world predict
another" is counted as **supported for an outcome** only if P beats both B1 and
B2 in held-out MAE by at least 10% *and* held-out Spearman rho >= 0.5 on the
leave-one-planet-out predictions. It is counted as **supported overall** only
if that holds for at least 4 of the 6 outcomes. Otherwise it is reported as
not supported (or partially supported, listing which outcomes).

I do not have a strong prior that this passes. In particular B2 is a strong
baseline because `tempo` is itself the staleness dial; P can only beat it
through planet-specific information carried by realised speed, measured blob
motion under the adversary, and the worth trajectory.

## Secondary, directional expectations (stated with their uncertainty)

1. *Coordination.* `V_comm >= 0` in most conditions. Direction with staleness is
   genuinely uncertain: stale remote information should lose value as `S`
   grows, but fresh shared information is also worth more when private memory
   decays. I expect a decline at the highest tempo; a non-monotone curve would
   not surprise me. On the `jam` planet I expect `V_comm` to be smaller than on
   `corrosion`/`quake` at matched `S`, because the channel itself is attacked.
2. *Coupling.* `V_couple` ordered escort > haul > search (worth concentration:
   one VIP vs. pairs vs. independent sweepers), largest on `quake` (rescue),
   and decreasing with `S`.
3. *Size.* `beta < 1` everywhere; smallest for escort; hazards that destroy
   robots (`corrosion`, `combined`) raise `N*` relative to `jam`.
4. *Sacrifice.* `V_sac > 0` mainly on `corrosion`/`combined` and ~0 on `jam`.
   In `default` policies on `corrosion`/`combined`, hazard exposure per live
   robot is higher in the last third of the mission than in the first third
   (residual worth has fallen), and across (condition x removal time) exposure
   correlates negatively with `W(t)`.

## Validity checks that gate interpretation

* Learned `default` policies must match or beat the scripted heuristic of the
  same mission in at least 9 of 12 cells (nature, N=6, tempo 0.3); cells that
  fail are flagged and excluded in a sensitivity analysis.
* The learned adversary must reduce `J` relative to nature for the `default`
  arm in most cells; a best-response adversary trained against the frozen
  `default` swarm is reported for at least one seed per cell as an
  exploitability check on the pool-minimum score.
* Seed-to-seed spread of each outcome is reported; an outcome whose seed spread
  exceeds its between-condition spread is flagged as unresolved.

## Known limitations fixed in advance

Abstract kinematic simulator; engineered egocentric observations; scripted VIP;
modest training budgets on 4 CPU cores; one-factor-at-a-time arms (no
interaction between comm, coupling and protection is estimated); policies are
trained over randomised tempo/N rather than per condition.

## Amendments made before any grid result existed

Recorded here so the history is honest; all were made after scripted-policy
balance sweeps and four throw-away 80-iteration training runs, and before the
grid was launched (see `docs/DESIGN.md` sections 8-9 and the git history).

1. Tempo grid changed from {0.3, 0.8, 1.3} to {0.1, 0.3, 0.6}; the heuristic
   validity check uses tempo 0.3. Reason: hazards faster than the robots they
   hunt made several cells unwinnable for any policy.
2. Constants `RHO_C`, `VIP_SPEED`, `HAUL_NORM` rebalanced and a jam "beacon"
   coupling mechanic added, for the same reason.
3. The adversary became a learned residual on a scripted pursuit prior, with a
   potential-based proximity shaping term in its training reward, because a
   from-scratch adversary did not learn to pressure the swarm in three rounds
   of smoke runs. The "learned adversary" in this study is therefore a learned
   correction to pursuit, not a from-scratch policy.
4. `tau_stale` is measured over 90-step windows starting at steps 0/30/60
   (capped at 90, not T) so that late starts are not censored differently.
5. Secondary expectation 1 ("decline at the highest tempo") now refers to
   tempo 0.6 vs 0.3.
6. Escort metric changed to time-averaged progress x health factor (the
   end-point metric saturated at 1 with 3x time slack), and in `search` the
   adversary observes the nearest unfound target. Both after smoke runs showed
   ceiling effects / an adversary with nothing to aim at.
