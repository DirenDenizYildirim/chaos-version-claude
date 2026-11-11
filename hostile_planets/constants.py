"""Simulator constants (see docs/DESIGN.md section 3)."""

T = 150            # episode horizon (steps)
N_MAX = 12         # robot slots
K = 3              # blob slots per hazard type
N_TYPES = 3        # 0 corrosion, 1 jam, 2 quake
NB = N_TYPES * K   # blob b has type b // K
V0 = 0.03          # robot top speed (arena units / step)

SIGMA = 0.09       # blob field f = amp*exp(-d^2/(2 SIGMA^2)), summed per type, clipped to [0,1]
RHO_C = 0.035      # health loss / step at corrosion f=1
RHO_AMB = 0.0015   # ambient health loss / step (corrosion + combined planets)
JAM_SENSE = 0.8    # sensing radius multiplier (1 - JAM_SENSE*f_jam)
P_LOSS_AMB = 0.05  # ambient packet loss
QUAKE_SLOW = 0.6   # speed multiplier (1 - QUAKE_SLOW*f_quake)
QUAKE_SLIP = 0.4   # slip std = QUAKE_SLIP*V0*f_quake
P_TRAP = 0.15      # per-step trap prob = P_TRAP*f_quake^2 (only while not stuck)
D_STUCK = 25       # steps immobilised
D_STUCK_VIP = 40
RESCUE_RATE = 8    # extra timer decrement / step with a free live teammate within R_LINK
TRAP_DAMAGE = 0.10

R_SENSE = 0.18     # blob-centre / teammate / VIP sensing radius
R_SENSE_VIP = 0.08 # the VIP's own (poor) blob sensing radius
R_DET = 0.05       # target detection radius (search)
R_VISIT = 0.045    # a grid cell counts as visited within this distance of its centre
R_COMM = 0.30      # gossip range
R_LINK = 0.06      # co-presence radius (rescue, joint carry, shielding)
R_CROWD = 0.025    # crowding: speed *= 1/(1+0.5*n_close)
R_ZONE = 0.07      # depot / goal / destination capture radius
R_SAFE = 0.20      # blob centres are projected out of this radius around anchors
BLOB_LO, BLOB_HI = 0.05, 0.95   # blob centres reflect at these walls
NATURE_TURN_STD = 0.3           # heading random-walk std (rad / step) for nature blobs

LOAD_SLOW = 0.6    # speed multiplier while carrying
VIP_SPEED = 0.45   # x V0
VIP_REP = 1.5      # VIP potential field: radial repulsion gain
VIP_TAN = 1.0      # VIP potential field: tangential (go-around) gain
VIP_MEM = 30.0     # VIP recency weight exp(-age/VIP_MEM)
M_TARGETS = 16
GRID = 8
HAUL_NORM = 40.0   # J_haul = units delivered / HAUL_NORM

# training-reward scales (the metric J never includes these)
DEATH_PEN = 1.0
SEARCH_REW = 4.0
HAUL_SHAPE = 2.0
ESCORT_REW = 10.0

# observation scaling
AGE_SCALE = 50.0   # ages are reported as clip(age/AGE_SCALE, 0, 1)
MATE_MAX_AGE = 60.0  # teammate beliefs older than this are ignored in the obs
REP_MEM = 30.0     # recency weight of the blob repulsion feature

MISSIONS = ("search", "haul", "escort")
COMM_MODES = ("silent", "gossip", "blackboard")
