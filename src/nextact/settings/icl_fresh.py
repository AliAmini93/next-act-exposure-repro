"""Prespecified fingerprints, model revisions and caps of the fresh-sample replication (fixed before any fresh-sample
output).
"""

FROZEN_AT = "2026-10-27"
STUDY = "a5r"
CORPORA = ("sgd", "ami")
DESIGN = {
    "sgd": "independent_cluster_replication (dialogue-disjoint from the first sample)",
    "ami": "within_corpus_robustness (unused targets of the same meetings and series as the first sample)",
}

FINGERPRINTS = {
    "sgd": {
        "queries": "373e52826cc0c83f10b5587f068068a1976292328807961b7e351ad8bb0e06d4",
        "plans": "38e0ce782f651d35da0ec2d3c005a8ce9dc14345a8d410442aa4c58d669ff1fa",
        "exp_f": "e150dfbc3942fcebe7044d773ab17e4c72e0652513b50bb9b0336004dc102ac4",
        "inventory": "f51f6adb1f9bfab27cef01b082d9f9c456726a40031acc978ddeb9c240ef0865",
        "data": "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea",
        "a5_queries": "c28ce1230de10b2a6bb7d9694a5460647df4ca641ae8ce127a70c3903dad52ea"
    },
    "ami": {
        "queries": "d1288c2dc3230d2faf29516f0f9f808f8fa2cd3c07bc3ef67933bfa82aa83d79",
        "plans": "309dcd2e308c5a91469b74a80ef80c274e244d9a4e826f4c53d4f1cdd2ed6991",
        "exp_f": "5a96c744bc05d8fbda5b042ca73780de5c457c1c80c1dc41813f5332ca430c11",
        "inventory": "4af7f7115150daea0beb62d362276e553d0864a69303a88be9892dc1cd5631d6",
        "data": "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd",
        "a5_queries": "c90f6fcf66c1ed0de152e6ca0dd3afe91e9e466bf18b24293e3181f7d549bc00"
    }
}

MODEL_REVISIONS = {
    "qwen25_7b": "a09a35458c702b33eeacc393d103063234e8bc28",   # Qwen/Qwen2.5-7B-Instruct, Apache-2.0, primary replication arm
    "mistral_7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",  # mistralai/Mistral-7B-Instruct-v0.3, Apache-2.0, second model
}

# one evaluation = one scored or generated prompt on real data (fresh queries); synthetic canaries excluded
EVALUATION_CAPS = {
    "qwen25_7b": 21000,   # projected 20,250: SGD 11 x 750 + 2 x 750 generation; AMI 12 x 750 + 2 x 750 generation
    "mistral_7b": 14000,  # projected 13,500: 9 order-A/B evaluations x 750 x 2 corpora
}
HALT_HOURS_PER_CORPUS = 8.0
MAX_PROMPT_TOKENS = 4096

