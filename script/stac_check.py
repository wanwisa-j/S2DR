# S2_SuperResolution/utils/stac_check.py
#
# CDSE auth helper: obtain a bearer token for the Copernicus Data Space
# OData catalog, shared by the modules that do their own metadata-only
# scene queries (daily_run.select_covered_cells, tile_grouping). No
# download here -- a metadata search request is cheap and separate from
# CDSE's download quota.
#
# Uses the same CDSE account as the Sentinel-2 pipeline (copied into this
# repo's own .env as CDS_USERNAME / CDS_PASSWORD -- see repository.py's
# load_dotenv).
#
# The per-bbox / per-tile "find latest scene" helpers that used to live
# here were removed 2026-09-07: the availability schedule that called them
# was retired 2026-08-19 and nothing has referenced them since. `git`-less
# tree, so the last copy is in the deadcode backup alongside this change.

import os
from typing import cast

import requests

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def _get_cdse_token() -> str:
    username = os.environ.get("CDS_USERNAME", "").strip()
    password = os.environ.get("CDS_PASSWORD", "").strip()
    if not username or not password:
        raise RuntimeError("CDS_USERNAME/CDS_PASSWORD not set in .env")

    resp = requests.post(
        _TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return cast(str, resp.json()["access_token"])
