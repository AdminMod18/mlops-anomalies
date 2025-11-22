#!/bin/bash

# MLOps Environment Validation Script

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     MLOps Environment Validation Script                      ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

check_ok() { echo -e "  ${GREEN}✅ OK${NC}     $1"; ((PASSED++)); }
check_fail() { echo -e "  ${RED}❌ ERROR${NC}  $1"; ((FAILED++)); }
check_warn() { echo -e "  ${YELLOW}⚠️ WARN${NC}   $1"; ((WARNINGS++)); }

echo -e "${BLUE}━━━ Python Environment ━━━${NC}"
if command -v python3 &> /dev/null; then
    version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    check_ok "Python: v$version"
else
    check_fail "Python: NOT INSTALLED"
fi

if command -v pip &> /dev/null; then
    check_ok "pip: installed"
else
    check_fail "pip: NOT INSTALLED"
fi

echo -e "\n${BLUE}━━━ Core Packages ━━━${NC}"
for pkg in scikit-learn pandas numpy fastapi uvicorn mlflow dvc prometheus-client; do
    if pip show $pkg &> /dev/null; then
        version=$(pip show $pkg 2>/dev/null | grep "Version" | cut -d: -f2 | tr -d ' ')
        check_ok "$pkg: v$version"
    else
        check_fail "$pkg: NOT INSTALLED"
    fi
done

echo -e "\n${BLUE}━━━ Container Tools ━━━${NC}"
if command -v docker &> /dev/null; then
    check_ok "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
else
    check_fail "Docker: NOT INSTALLED"
fi

if docker info &> /dev/null; then
    check_ok "Docker daemon: running"
else
    check_warn "Docker daemon: NOT RUNNING"
fi

echo -e "\n${BLUE}━━━ Kubernetes Tools ━━━${NC}"
if command -v kubectl &> /dev/null; then
    check_ok "kubectl: installed"
else
    check_fail "kubectl: NOT INSTALLED"
fi

if command -v minikube &> /dev/null; then
    check_ok "Minikube: installed"
    status=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "Stopped")
    if [ "$status" == "Running" ]; then
        check_ok "Minikube status: running"
    else
        check_warn "Minikube status: $status"
    fi
else
    check_fail "Minikube: NOT INSTALLED"
fi

echo -e "\n${BLUE}━━━ Version Control ━━━${NC}"
if command -v git &> /dev/null; then
    check_ok "Git: $(git --version | cut -d' ' -f3)"
else
    check_fail "Git: NOT INSTALLED"
fi

echo -e "\n${BLUE}━━━ Project Files ━━━${NC}"
for file in requirements.txt Dockerfile Makefile dvc.yaml config/config.yaml; do
    if [ -f "$file" ]; then
        check_ok "File: $file"
    else
        check_warn "File missing: $file"
    fi
done

if [ -f "models/isolation_forest_model.joblib" ]; then
    check_ok "Model: trained"
else
    check_warn "Model: NOT TRAINED (run 'make train')"
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}                      SUMMARY                                 ${BLUE}║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC}   ${GREEN}✅ Passed:${NC}   $PASSED                                            ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}   ${RED}❌ Failed:${NC}   $FAILED                                            ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}   ${YELLOW}⚠️  Warnings:${NC} $WARNINGS                                            ${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${YELLOW}To fix missing packages: pip install -r requirements.txt${NC}"
    exit 1
fi
