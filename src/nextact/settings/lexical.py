"""Prespecified sample sizes, data and plan fingerprints and fit counts of the randomised presence design
(fixed before any model fit). A run whose data or plan fingerprint differs is a replication on different data, not
the original sample.
"""

SAMPLE_N = {"sgd": 3000, "star": 3000, "ami": None}

FINGERPRINTS = {
    'sgd': {
        'data': "81d6c5aaf8e69d55e0d5d54e2e5e46ebdb84fb46d8ea85e654a1b1feaf19fdea",
        'dialogues': "13eafe7a65407818844c50f58ce648c854aecda9362eeec783bdbc5d3f29e54d",
        'D1': "74bcb3cdce24897753f1f176aff9844b69997907991fc8b3e7532dcb87cf1670",
        'D2': "8617e392445014a22997bf1553b9ed774347b81a8fe6619a11922d62aec0c361",
        'D3': "20f9c91bd2bcc6387603edd70605450784bfbc9117c38c836311414beff09805",
        'D3_groups': 12,
        'D6': "07c7bc286b2a89e2bec94b0c9bc15064e60642de27800864e656b6af4bad794f",
    },
    'star': {
        'data': "83081e28c4cf44a3c35e38ac0ff26304ccceac050e7ebcce167f885656e0151b",
        'dialogues': "aa4e9d45b78cc2beb10aee037eac2b94f9654e45a88f043460ef998530f68cc6",
        'D1': "6778aa1600e06807116620f5a639484236c6f3efbd5c63798ae7555b99512de5",
        'D2': "46faedf3e11bc99a1b8229a70a9d01abaf7c38a7017c357255b2adde1585891e",
        'D3': "c65bacb980940045ce6e4d7ccd73dd11e4db4c93230d43852ad9b5da9d61aa07",
        'D3_groups': 12,
        'D6': "f3432994f80f613a4b1e661e2c134d00802a1f33f4db9a88d025ff93e37d38dc",
    },
    'ami': {
        'data': "7bf1d5e8b74e7a514214ad9561ecc36962f4e18ff175350a53060fb7291ed5fd",
        'dialogues': "142a9335f8b3fc0928e84f6eebb05a7b2992410a46b9860c36ea0a1226be67b5",
        'D1': "a5ea9848f4f5552580eaf4222e009432cca6dfb958ae8a1ec87552ff0a21fab3",
        'D2': "49d8acf4d9f0ddbe1f0e0575e82dcc49d5763a463282507d29076a833b20dfa5",
        'D3': "9d8fa97c34efaf40bfcc8c4295e5a7782aa5cc2cf02fdcbd166ddb96265cd418",
        'D3_groups': 24,
        'D6': "2a08b41a724f3c5948282675f9230ab87297a64be0395908139ab6cc88e22f0c",
    },
}

N_FITS = {
    'sgd': {"D1": 125, "D2": 625, "D5": 100, "D6": 25, "D3": 185, "D4": 80},
    'star': {"D1": 125, "D2": 625, "D5": 100, "D6": 25, "D3": 185, "D4": 80},
    'ami': {"D1": 125, "D2": 625, "D5": 100, "D6": 25, "D3": 365, "D4": 80},
}
