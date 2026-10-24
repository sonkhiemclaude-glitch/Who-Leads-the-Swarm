"""Parameter values stated in the paper (Table 2 and Sections 4-5).

Every value here is referenced by section so that configs can override it explicitly.
"""
from __future__ import annotations

# --- Radio and energy model (Section 5.3, Table 2: ns-3 WifiRadioEnergyModel defaults) ---
SUPPLY_VOLTAGE_V = 3.0
I_TX_A = 0.380
I_RX_A = 0.313
I_IDLE_A = 0.273
PHY_RATE_BPS = 6e6  # IEEE 802.11a OfdmRate6Mbps
IP_MTU_B = 1500
IP_UDP_HEADER_B = 28  # IPv4 (20) + UDP (8), used for the fragment count
PER_FRAGMENT_OVERHEAD_B = 8 + 20 + 8 + 28  # UDP + IPv4 + LLC/SNAP + 802.11 header and FCS
PREAMBLE_S = 20e-6
MAC_ACK_S = 44e-6
SECURITY_OVERHEAD_B = 29  # DTLS-style record header + AEAD tag (Section 3.1)

# --- Scenario (Table 2) ---
AREA_M = 800.0
ALTITUDE_M = (50.0, 150.0)
MISSION_S = 600.0
WARMUP_S = 10.0
TX_POWER_DBM = 20.0
PATH_LOSS_EXPONENT = 2.2
REFERENCE_LOSS_DB = 46.6777
RANGE_ESTIMATE_M = 250.0  # R-hat (Section 5.4)
DEFAULT_MONITORS = ((200.0, 200.0, 30.0), (600.0, 200.0, 30.0), (200.0, 600.0, 30.0), (600.0, 600.0, 30.0))
EDGE_MONITOR = ((0.0, 400.0, 150.0),)

# --- Gauss-Markov mobility (Table 2) ---
GM_MEMORY = 0.85
GM_SPEED_SIGMA = 3.0
GM_HEADING_SIGMA = 0.4
GM_PITCH_SIGMA = 0.05
GM_EDGE_M = 100.0
GM_SPEED_CLIP = (0.2, 2.0)
RPGM_RADIUS_M = 250.0
RPGM_DRIFT_MPS = 3.0

# --- Mission script (Section 4.1, Table 2) ---
PHASES = ("search", "track", "regroup", "return")
BASE_REPORT_PERIOD_S = {"search": 10.0, "track": 4.0, "regroup": 6.0, "return": 15.0}
BASE_REPLAN_RATE_HZ = {"search": 1 / 120, "track": 1 / 50, "regroup": 1 / 90, "return": 0.0}
PROFILE_RANGE = (0.6, 1.6)
PHASE_DURATION_S = (60.0, 140.0)
RETURN_PHASE_S = 40.0

# --- Protocol (Section 5) ---
ROTATION_DELAY_S = (4.0, 20.0)
ROTATION_TOP_K = 3
DEFAULT_SIZE_PROFILE_B = {"plan": 900, "ack": 260, "report": 600}
SIZE_MEMORY_MIN_SAMPLES = 5
BURST_GAP_S = (0.02, 0.15)
ACK_DELAY_S = (0.2, 1.0)
DECOY_DELIVERY_ASSUMPTION_S = 0.05
DECOY_START_S = (0.3, 2.5)
TARGETED_SHARES = {"cotimed": 0.4, "spread": 0.3, "mirror": 0.3}
N_COTIMED_DECOYS = 2
TOKEN_BUCKET_CAPACITY = 25.0
TOKEN_BUCKET_RATE_HZ = 3.0
LINK_GAP_S = 0.5
PAD_CHUNK_B = 1024
CONST_COVER_SIZE_B = 1024
BUDGETS_J = (0.05, 0.2, 1.0)

# --- Adversary (Sections 4.3-4.4, Table 2) ---
SIZE_BINS_B = (0.0, 200.0, 500.0, 1000.0, 1400.0, 1e9)
LARGE_FRAME_B = 1400
SMALL_FRAME_B = 500
ROLE_WINDOW = (10.0, 5.0)  # (W, step)
PHASE_WINDOW = (20.0, 5.0)
PLAN_WINDOW = (4.0, 1.0)
HORIZONS_S = (0.0, 10.0, 30.0)
PLAN_TOLERANCE_S = 5.0
PLAN_THRESHOLD = 0.5
PLAN_MERGE_S = 3.0
RF_TREES = 200
RF_MIN_LEAF = 2
CV_FOLDS = 4
