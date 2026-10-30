"""Prespecified fingerprints, model revisions and checks of the unlabelled-context control.
"""

FROZEN_AT = "2026-10-28"
STUDY = "g8"
ROLE = "exploratory control motivated by the fresh-sample wrong-label (exp_f) outcome; changes no prespecified decision"
CORPORA = ("sgd", "ami")
N_PER_CELL = {"continuation": 150, "transition": 150}
N_LABELLED = 3
ARMS = ("l3", "u_same", "u_other", "g_same", "w_same")
PRIMARY_CONTRAST = ("u_same", "u_other", "continuation", "logscore")

FINGERPRINTS = {
    "sgd": {
        "queries": "7184a61e0b65912489622409beae31d2345221e93825e2343317b46766505f87",
        "plan": "a2297987493f4419e1e024b530bbfa2b188e61c788c4467fb629961b128edc67",
        "inventory": "f51f6adb1f9bfab27cef01b082d9f9c456726a40031acc978ddeb9c240ef0865",
        "data": "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea",
        "a5r_queries": "373e52826cc0c83f10b5587f068068a1976292328807961b7e351ad8bb0e06d4"
    },
    "ami": {
        "queries": "993d0b6072e16909014da7257c48624c0499b50fc0d9d3648412d5f90fee6c06",
        "plan": "17ea38f5ddb02b51f6e18d491dc4943a592a9e356d0e46223ede4a296ad87b7e",
        "inventory": "4af7f7115150daea0beb62d362276e553d0864a69303a88be9892dc1cd5631d6",
        "data": "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd",
        "a5r_queries": "d1288c2dc3230d2faf29516f0f9f808f8fa2cd3c07bc3ef67933bfa82aa83d79"
    }
}

MODEL_REVISIONS = {
    "qwen25_7b": "a09a35458c702b33eeacc393d103063234e8bc28",   # Qwen/Qwen2.5-7B-Instruct, Apache-2.0, run first
    "mistral_7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",  # mistralai/Mistral-7B-Instruct-v0.3, Apache-2.0, second
}

# one evaluation = one scored prompt on real data; 5 arms x 300 queries x 2 corpora = 3,000 per model
EVALUATION_CAP_PER_MODEL = 3200
TOTAL_CAP = 6400
HALT_HOURS_PER_CORPUS = 2.0
MAX_PROMPT_TOKENS = 4096
TOKEN_GATE = 25
BOOTSTRAP = dict(n_boot=2000, seed=777)

