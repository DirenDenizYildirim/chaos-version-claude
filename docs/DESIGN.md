# Hostile Planets — design spec

Abstract 2-D kinematic simulator (not a physics sim) for ground-robot swarms on
hostile "planets", written in JAX so whole episodes are `jit`/`vmap`/`scan`-able
on CPU. This file is the contract between the simulator, the trainer and the
analysis code.

## 1. Research questions and how each is operationalised

| Question | Experimental arm | Outcome |
|---|---|---|
| How should the swarm coordinate? | `comm` in {`silent`, `gossip`, `blackboard`} | `V_comm = (J[blackboard] - J[silent]) / J_ref`, also gossip |
| When rely on tight coupling? | `coupling` in {on, off}; off disables all co-presence synergies | `V_couple = (J[on] - J[off]) / J_ref`; usage rate of coupled states |
| How large should it be? | `n_robots` in {2,3,4,6,8,12}, one policy trained over random N | `J(N)`, marginal worth, `N*(lambda) = argmax J(N) - lambda N` |
| When sacrifice a member? | `protect` in {off, on}; on adds a death penalty in training | price of safety `V_sac = (J[free] - J[protected]) / J_ref`; loss timing/health |

Everything is relative to a stated mission metric `J` (section 5).

Principle variables, *measured* not assumed:

* **Staleness ratio** `S = tau_act / tau_stale`. `tau_stale` = mean time for a
  hazard blob centre to move more than `SIGMA` from where it was observed
  (measured from blob trajectories). `tau_act` = mission length scale / mean
  realised robot speed.
* **Worth trajectory** `W(t)` = drop in final `J` when one robot is force-removed
  at step `t` (counterfactual rollouts), normalised by `J/N`.

## 2. Static layout

* Arena `[0,1]^2`, horizon `T = 150` steps, fixed-length episodes (no early reset;
  finished things freeze).
* `N_MAX = 12` robot slots; `n_robots` per env is dynamic (extra slots absent).
* 3 hazard types (0 corrosion, 1 jam, 2 quake) x `K = 3` blob slots = 9 blobs.
  Per-env amplitude vector `amp[9]` (0 = slot unused).
* Missions are *static* (one compiled function per mission):
  `search`, `haul`, `escort`.

## 3. Constants (`hostile_planets/constants.py`)

```
T=150  N_MAX=12  K=3  NB=9  V0=0.03
SIGMA=0.09            # blob field f = amp*exp(-d^2/(2 SIGMA^2)), summed per type, clipped to [0,1]
RHO_C=0.035           # health loss / step at corrosion f=1
RHO_AMB=0.0015        # ambient health loss / step (param `ambient`, on for corrosion+combined)
JAM_SENSE=0.8         # sensing radius multiplier (1 - JAM_SENSE*f_jam)
P_LOSS_AMB=0.05       # ambient packet loss
QUAKE_SLOW=0.6        # speed multiplier (1 - QUAKE_SLOW*f_quake)
P_TRAP=0.15           # per-step trap prob = P_TRAP*f_quake^2 (only while not stuck)
D_STUCK=25            # steps immobilised; D_STUCK_VIP=40
RESCUE_RATE=8         # extra timer decrement / step if a free live teammate within R_LINK (coupling on)
TRAP_DAMAGE=0.10
R_SENSE=0.18          # blob-centre / teammate / VIP sensing radius
R_DET=0.05            # target detection radius (search)
R_COMM=0.30           # gossip range
R_LINK=0.06           # co-presence radius (rescue, joint carry, shielding)
R_CROWD=0.025         # crowding: speed *= 1/(1+0.5*n_close)
R_ZONE=0.07           # depot / goal / destination capture radius
R_SAFE=0.20           # blob centres are projected out of this radius around anchors
LOAD_SLOW=0.6         # speed multiplier while carrying
VIP_SPEED=0.45        # x V0
M_TARGETS=16  GRID=8
```

Robot speed cap: `V0*(0.4+0.6h)*(1-QUAKE_SLOW*f_q)*load*crowd`. Dead iff `h<=0`.

## 4. Hazards

* **Corrosion** attacks the body: `dh = -(ambient + RHO_C*f_c)`; health lowers speed.
* **Jam** attacks sensing/comms: the commanded velocity is rotated by
  `theta ~ U(-pi,pi)*f_j` (nav error); sensing/detection radii shrink by
  `(1-JAM_SENSE*f_j)`; a packet i<-j is delivered w.p.
  `(1-P_LOSS_AMB)(1-f_j(x_i))(1-f_j(x_j))`.
* **Quake** attacks mobility: slow-down, random slip `N(0,(0.4*V0*f_q)^2)`,
  trapping (stuck `D_STUCK` steps, `TRAP_DAMAGE` health).
* Blob motion: `c_k += tempo*V0*u_k`, `|u_k|<=1`. In *nature* envs `u_k` is a
  persistent random heading (heading random-walk, std 0.3 rad/step). In
  *adversarial* envs `u_k` is the adversary's action. Centres reflect at
  `[0.05,0.95]` and are projected out of `R_SAFE` around mission anchors.
  `tempo` is the staleness dial.

Planet presets (`planets.py`), amplitudes per type (K slots each):

| planet | corrosion | jam | quake | ambient |
|---|---|---|---|---|
| `corrosion` | 1,1,1 | 0 | 0 | on |
| `jam` | 0 | 1,1,1 | 0 | off |
| `quake` | 0 | 0 | 1,1,1 | off |
| `combined` | .8,.8,0 | .8,.8,0 | .8,.8,0 | on |

## 5. Missions and metric J

Anchors: a *left* anchor `x~U(.1,.2), y~U(.2,.8)` and a *right* anchor
`x~U(.8,.9), y~U(.2,.8)`.

* **search** — robots start clustered at the left anchor (safe zone). 16 static
  targets uniform in the arena. A target is found when a live robot is within
  `R_DET*(1-JAM_SENSE*f_j(robot))`. `J = (1/M) * sum_found (1 - t_found/T)`.
  Each robot holds an 8x8 visited-cell belief (OR-merged by comm).
* **haul** — depot = left anchor, goal = right anchor, robots start at depot.
  A non-carrying robot inside the depot zone with engage `e<=0` picks a solo
  crate (1 unit). With `e>0` (and coupling on) it waits for a mutual-nearest
  partner within `R_LINK` that also has `e>0`; the pair lifts a heavy crate
  (3 units total). A pair is rigidly coupled: both sit at the pair position, the
  pair moves with the mean of both commands at the min of both speed caps. The
  pair breaks (crate lost) if either dies, gets stuck, or sets `e<=0`.
  Delivery inside the goal zone. `J = units_delivered / 40`.
* **escort** — a scripted VIP (speed `VIP_SPEED*V0`, health 1, sensing radius
  0.08) travels left anchor -> right anchor with a potential-field controller
  that avoids blobs *in its own belief table* (recency weighted). The VIP is a
  comm node. Escorts help by relaying blob sightings (comm), by rescuing a
  trapped VIP (within `R_LINK`, coupling on), and by shielding: each escort
  within `R_LINK` with `e>0` (max 2) halves the VIP's corrosion dose and takes
  the diverted dose itself. `J = progress * (0.3 + 0.7*h_vip)` at T, progress =
  1 - remaining/initial distance (1 on arrival; VIP safe once arrived; frozen if dead).

Training reward per robot = `0.5*individual + 0.5*team` for search/haul (plus
potential-based distance shaping for haul), pure team `dPhi` for escort. In the
`protect` arm every death costs `DEATH_PEN` to the dying robot and the team.
The *metric* `J` never includes shaping or the death penalty.

## 6. Information model

Per robot belief tables with explicit timestamps:
`blob_pos[NB,2], blob_t[NB]`, `mate_pos[N_MAX,2], mate_t[N_MAX], mate_h[N_MAX]`,
mission table (visited grid / VIP pos+time). Own sensing refreshes entries within
range. Comm merges tables entry-wise by newest timestamp.

* `silent` (0): no comm.
* `gossip` (1): pairwise merge between robots within `R_COMM`, packet loss as above.
* `blackboard` (2): a hub table; each robot uplinks/downlinks w.p.
  `(1-P_LOSS_AMB)(1-f_j(x_i))` regardless of range.

Robots always read in-situ field value and gradient of all three hazard types.

## 7. API (`hostile_planets/env.py`)

```python
env = HostileEnv(mission="search"|"haul"|"escort")        # static
env.obs_dim, env.act_dim (=3), env.adv_obs_dim, env.adv_act_dim (=2)
p = env.make_params(planet="corrosion", n_robots=8, tempo=0.6, comm="gossip",
                    coupling=True, adversarial=False, protect=False,
                    remove_t=-1)          # NamedTuple of jnp scalars/arrays, vmappable
state, obs = env.reset(key, p)
state, obs, rew, done, info = env.step(key, state, robot_act, adv_act, p)
```

* `robot_act [N_MAX,3]` in `[-1,1]` (vx, vy, engage); `adv_act [NB,2]`.
* `obs = {"robot":[N_MAX,obs_dim], "robot_mask":[N_MAX] (alive), "adv":[NB,adv_obs_dim], "adv_mask":[NB] (amp>0)}`.
* `rew = {"robot":[N_MAX], "adv": scalar (= -team metric increment)}`.
* `done` scalar bool, true at `t == T`.
* `info`: scalars `J` (cumulative metric so far), `n_alive`, `deaths` (cumulative),
  `coupled_frac` (fraction of live robots in a coupled state this step),
  `exposure` (mean corrosion+quake field over live robots), `mean_speed`,
  `mean_health`, plus mission extras.
* `remove_t >= 0`: at that step the lowest-index live robot is removed (counterfactual worth).
* Everything is pure and `vmap`-able over `(key, state, acts, params)`.

Observations are egocentric, engineered, roughly in `[-1,1]`: self state,
in-situ fields + gradients, nearest believed blob per type (rel pos, age, known
flag) and a recency-weighted repulsion vector per type, two nearest known
teammates + centroid, and a mission block.

Adversary obs per blob: own pos, type one-hot, rel pos to the two nearest live
robots, to swarm centroid, to mission anchors/VIP, `t/T`.
