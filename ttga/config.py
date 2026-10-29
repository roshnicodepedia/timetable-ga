"""Global constants of the timetable problem (change here, everything follows)."""

DAYS = 5
SLOTS_PER_DAY = 6
N_SLOTS = DAYS * SLOTS_PER_DAY
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# cost = W_HARD * (number of hard violations) + weighted soft penalties
W_HARD = 10000
W_GAP = 1        # S1: idle gaps inside a batch's day
W_SAMEDAY = 2    # S2: same course twice on one day
W_CONSEC = 1     # S3: a teacher teaching more than MAX_CONSEC slots in a row
W_BALANCE = 1    # S4: uneven daily load of a batch
MAX_CONSEC = 3
