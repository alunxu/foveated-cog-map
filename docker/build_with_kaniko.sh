#!/bin/bash
# Build docker/Dockerfile using kaniko on RCP, push to registry.rcp.epfl.ch.
#
# Kaniko runs inside a kubernetes pod and builds Docker images without needing
# privileged access or a Docker daemon. RCP supports it.
#
# Usage:
#   bash docker/build_with_kaniko.sh [TAG]
# Default TAG: v1

set -e

TAG="${1:-v1}"
REGISTRY="registry.rcp.epfl.ch"
PROJECT="dhlab-wxu"
IMAGE_NAME="habitat"
DEST="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${TAG}"

echo "=================================================="
echo "  Build + push Docker image via kaniko"
echo "  Image: ${DEST}"
echo "  Build context: $(pwd)/docker/"
echo "=================================================="

# Ensure we're at the project root
if [ ! -f docker/Dockerfile ]; then
    echo "ERROR: must be run from project root (where docker/Dockerfile lives)"
    exit 1
fi

# Tar the build context (Dockerfile + any small files it needs) — keep it minimal
TARBALL=/tmp/dh-spatial-docker-context.tar.gz
tar czf $TARBALL --exclude='.git' --exclude='node_modules' docker/

# Push tarball into a temporary kaniko build pod via stdin

# Approach: use a kaniko pod with the tarball mounted via configmap or via init container
# Simplest path: copy tarball to /scratch/wxu/dh-spatial/docker-context.tar.gz, then kaniko reads from there

echo "[1/3] Copying build context to scratch (via setup pod)..."
POD=$(kubectl get pod -n runai-dhlab-wxu -l runai/job-name=dh-spatial-setup -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$POD" ]; then
    echo "ERROR: dh-spatial-setup pod not running on RCP. Submit it first."
    exit 1
fi

cat $TARBALL | kubectl exec -i -n runai-dhlab-wxu $POD -- bash -c "
mkdir -p /scratch/wxu/dh-spatial-docker-context
cd /scratch/wxu/dh-spatial-docker-context
tar xzf - && ls
"

echo "[2/3] Submitting kaniko build pod..."
RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod submit dh-spatial-kaniko \
    --image=gcr.io/kaniko-project/executor:latest \
    --gpu=0 --cpu=4 --memory=16G \
    --pvc=dhlab-scratch:/scratch \
    --command -- /kaniko/executor \
        --dockerfile=/scratch/wxu/dh-spatial-docker-context/docker/Dockerfile \
        --context=dir:///scratch/wxu/dh-spatial-docker-context/docker \
        --destination="${DEST}" \
        --cache=true \
        --cache-repo="${REGISTRY}/${PROJECT}/habitat-cache"

echo ""
echo "[3/3] Watch build:"
echo "  RUNAI_CURRENT_CTX=rcp /usr/local/bin/runai-rcp-prod logs dh-spatial-kaniko --follow"
echo ""
echo "Build typically takes 15-30min (apt + conda installs)."
echo "Once it's done, use the image in submit:"
echo "  --image=${DEST}"
