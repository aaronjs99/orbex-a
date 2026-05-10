#!/usr/bin/env python3
# /***********************************************************
# *                                                         *
# * Copyright (c) 2026                                      *
# *                                                         *
# * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
# * University of California, Los Angeles                   *
# *                                                         *
# * Authors: Aaron John Sabu, Brett T. Lopez                *
# * Contact: {aaronjs, btlopez}@ucla.edu                    *
# *                                                         *
# ***********************************************************/

"""Console entrypoint for ADTMPC mission artifact generation."""

import os
from pathlib import Path

_mpl_cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/orbexa-matplotlib"))
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

from orbexa.visualization.demo import main


if __name__ == "__main__":
    main()
