"""Prespecified fingerprints, model revisions, caps and the dense model files of the follow-up experiments.
"""

FROZEN_AT = "2026-10-28"
STUDY = "g9p5"
ROLE = "follow-up exploratory interventions and dense-retriever prevalence; changes no prespecified decision"

# Experiment A
MECHANISM_CORPORA = ("sgd", "ami")
N_PER_CELL = {"continuation": 100, "transition": 100}
SELECTION_SALT = "G9P5|mechanism|20261028"
PLACEBO_SALT = "G9P5|placebo|20261028"
ARMS = ('none', 'overlap', 'older', 'full', 'other_overlap', 'other_older')
PRIMARY = {"S_overlap": ("overlap", "other_overlap", "all"), "S_history": ("older", "other_older", "nonempty_h"),
           "S_full": ("full", "none", "all")}
MEAN_TOKEN_GATE = 25
MEDIAN_TOKEN_GATE = 20
BOOTSTRAP = dict(n_boot=2000, seed=20261028)
MECHANISM_FINGERPRINTS = {
    "ami": {
        "a5r_queries": "d1288c2dc3230d2faf29516f0f9f808f8fa2cd3c07bc3ef67933bfa82aa83d79",
        "data": "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd",
        "g8_queries": "993d0b6072e16909014da7257c48624c0499b50fc0d9d3648412d5f90fee6c06",
        "inventory": "4af7f7115150daea0beb62d362276e553d0864a69303a88be9892dc1cd5631d6",
        "plan": "85b3ce1e7c7c2e0bdf9828a46623572d2e88b9b71b6a3017fa3164730af41540",
        "queries": "eb037074d80a720e3c38c76f434d43119d4d0fef00f1e9b7c7f727d6bfa26b42"
    },
    "sgd": {
        "a5r_queries": "373e52826cc0c83f10b5587f068068a1976292328807961b7e351ad8bb0e06d4",
        "data": "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea",
        "g8_queries": "7184a61e0b65912489622409beae31d2345221e93825e2343317b46766505f87",
        "inventory": "f51f6adb1f9bfab27cef01b082d9f9c456726a40031acc978ddeb9c240ef0865",
        "plan": "9542585cf7a7b398deebdd22f5fbc18b2eb3c225db3aaef0616a2d8089e564e4",
        "queries": "c3add89f0d5fcfea5a3ca4a34029b6a74b92c8725f4f55031412df49fc75a884"
    }
}
MODEL_REVISIONS = {
    "qwen25_7b": "a09a35458c702b33eeacc393d103063234e8bc28",   # Qwen/Qwen2.5-7B-Instruct, Apache-2.0, primary (run)
    "mistral_7b": "c170c708c41dac9275d15a8fff4eca08d52bab71",  # mistralai/Mistral-7B-Instruct-v0.3 (not run, see below)
}
MISTRAL_STATUS = "NOT_RUN_ENV_CONSTRAINT"
MISTRAL_DETAIL = ("tokenizer of the pinned revision requires the sentencepiece package, absent from the fixed interpreter; "
                  "installation forbidden by the protocol")
EVALUATION_CAP_PER_MODEL = 2500
TOTAL_CAP = 5000
PLANNED_EVALUATIONS = 2400
HALT_HOURS_PER_CORPUS = 2.0
MAX_PROMPT_TOKENS = 4096

# Experiment B
DENSE_MODEL = dict(repo="BAAI/bge-small-en-v1.5", revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", licence="MIT",
                   pooling="cls", normalise="l2", dim=384, max_tokens=512,
                   combined_sha256="874f582ac5efcdd2f28951559ed53f8da72a369a5820ac070bdd06323c3e11c5",
                   files={
    "1_Pooling/config.json": {
        "bytes": 190,
        "sha256": "d1caf60c96f5fba2157c0c26b76d80818fad6cf0b8eb5e73ec372ff9818eba5c"
    },
    "README.md": {
        "bytes": 94783,
        "sha256": "ddb964361a55c6e5dfca6361615854b260c9c960205d04c7520151aaa1d75837"
    },
    "config.json": {
        "bytes": 743,
        "sha256": "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750"
    },
    "model.safetensors": {
        "bytes": 133466304,
        "sha256": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
    },
    "modules.json": {
        "bytes": 349,
        "sha256": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf"
    },
    "special_tokens_map.json": {
        "bytes": 125,
        "sha256": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3"
    },
    "tokenizer.json": {
        "bytes": 711396,
        "sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    },
    "tokenizer_config.json": {
        "bytes": 366,
        "sha256": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3"
    },
    "vocab.txt": {
        "bytes": 231508,
        "sha256": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
    }
})
DENSE_DATA_FINGERPRINTS = {
    "ami": "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd",
    "sgd": "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea",
    "star": "83081e28c4cf44a3c35e38ac0ff26304ccceac050e7ebcce167f885656e0151b"
}
DENSE_QUERY_FINGERPRINTS = {
    "ami_a5": "c90f6fcf66c1ed0de152e6ca0dd3afe91e9e466bf18b24293e3181f7d549bc00",
    "ami_a5r": "d1288c2dc3230d2faf29516f0f9f808f8fa2cd3c07bc3ef67933bfa82aa83d79",
    "sgd_a5": "c28ce1230de10b2a6bb7d9694a5460647df4ca641ae8ce127a70c3903dad52ea",
    "sgd_a5r": "373e52826cc0c83f10b5587f068068a1976292328807961b7e351ad8bb0e06d4",
    "star_a5": "3c645488d8a070a1b59bf5851afc1b9eca13a514c550e0974c7362e5c6fdb07d"
}
TFIDF_CANARY = {"sgd_a5": (0.976, 698), "sgd_a5r": (0.975, 698), "ami_a5": (0.993, 716), "ami_a5r": (0.992, 711),
                "star_a5": (0.756, 504)}
DENSE_GATES = dict(max_truncation_share=0.05, high=0.75, low=0.25, gpu_cap_gib=16.0,
                   halt_hours=3.0)

