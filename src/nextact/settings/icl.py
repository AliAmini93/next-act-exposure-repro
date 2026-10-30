"""Prespecified fingerprints, model revisions and evaluation caps of the first-sample retrieval study (fixed before
any model output).
"""

FROZEN_AT = "2026-10-27"
PRIMARY_CORPORA = ("sgd", "ami")
OPTIONAL_CORPORA = ("star",)

FINGERPRINTS = {
    "sgd": {
        "queries": "c28ce1230de10b2a6bb7d9694a5460647df4ca641ae8ce127a70c3903dad52ea",
        "plans": "7ed38b4b336dde43f531aa5049daa9f518de8141535662867af8d47ebd7792f3",
        "inventory": "f51f6adb1f9bfab27cef01b082d9f9c456726a40031acc978ddeb9c240ef0865",
        "data": "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea"
    },
    "ami": {
        "queries": "c90f6fcf66c1ed0de152e6ca0dd3afe91e9e466bf18b24293e3181f7d549bc00",
        "plans": "25eac9f0f9b52366d9b4cf01d0046c2de0efaad930777c400e547a85b98fe0c6",
        "inventory": "4af7f7115150daea0beb62d362276e553d0864a69303a88be9892dc1cd5631d6",
        "data": "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd"
    },
    "star": {
        "queries": "3c645488d8a070a1b59bf5851afc1b9eca13a514c550e0974c7362e5c6fdb07d",
        "plans": "a826e9b9a97f8a67057ec06627bd8958a49a1ec25916b3ac05fac3b2f5dc0349",
        "inventory": "9bfa9701878f264fb93f11eb4ed9f87cf5fb2bf9a9d12898fabd30a00118c712",
        "data": "83081e28c4cf44a3c35e38ac0ff26304ccceac050e7ebcce167f885656e0151b"
    }
}

MODEL_REVISIONS = {
    "qwen25_7b": "a09a35458c702b33eeacc393d103063234e8bc28",   # Qwen/Qwen2.5-7B-Instruct, Apache-2.0, primary
    "mistral_7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",  # mistralai/Mistral-7B-Instruct-v0.3, Apache-2.0, optional
}

# evaluation caps (one evaluation = one scored or generated prompt on real data; synthetic canaries excluded)
EVALUATION_CAPS = {
    "qwen25_7b": 20000,   # primary: 15,750 scoring + 3,000 generation projected
    "mistral_7b": 9000,   # optional exploratory model-dependence arm: 6 evaluations x 750 queries x 2 corpora
    "star_qwen25_7b": 5000,  # optional exploratory STAR arm: 6 evaluations x 750 queries
}
HALT_HOURS_PER_CORPUS = 8.0
MAX_PROMPT_TOKENS = 4096

