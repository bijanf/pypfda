#!/usr/bin/env bash
# Download the Zenodo data deposit referenced by the paper-reproduction
# scripts. The DOI is set on the day the deposit is minted; until then
# this script just prints instructions.

set -euo pipefail

ZENODO_DOI="${PYPFDA_ZENODO_DOI:-}"

if [[ -z "${ZENODO_DOI}" ]]; then
    cat <<'EOF'
The Zenodo data deposit DOI has not yet been wired into this script.

Once minted, set the environment variable PYPFDA_ZENODO_DOI to the DOI
suffix (e.g. 10.5281/zenodo.XXXXXXX) and rerun:

    PYPFDA_ZENODO_DOI=10.5281/zenodo.XXXXXXX bash download_data.sh
EOF
    exit 0
fi

mkdir -p data
echo "Downloading from https://doi.org/${ZENODO_DOI} into data/ ..."
# Placeholder: real implementation would use zenodo-get or the REST API.
echo "(Implementation pending; v0.2.0)"
